from collections import defaultdict
from typing import Optional, List, Dict

from scripts.ensemble import EnsembleChoir, Ensembler
from scripts.utils import ENTITIES, RELATIONS, Collection

# =======================================================================
# - VotingEnsembler -----------------------------------------------------
# =======================================================================


class VotingEnsembler(Ensembler):
    def __init__(self, choir, orchestrator, weighter, scorer, validator):
        super().__init__(choir, orchestrator)
        self._weighter = weighter
        self._scorer = scorer
        self._validator = validator

    @property
    def weight(self):
        return self._weighter

    @property
    def score(self):
        return self._scorer

    @property
    def validate(self):
        return self._validator

    def _score_label(self, annotation, sid: int, label: str, votes: dict) -> float:
        weights = {submit: self.weight(label, sid, submit) for submit in votes}
        return self.score(weights, annotation, sid, label)

    def _validate_label(
        self, annotation, sid: int, label: str, score: float
    ) -> Optional[str]:
        return self.validate(annotation, sid, label, score)


# =======================================================================
# - Weighter ------------------------------------------------------------
# =======================================================================


class Weighter:
    def __call__(self, label: str, sid: int, submit: str) -> float:
        raise NotImplementedError()


class TableWeighter(Weighter):
    def __init__(self, table={}):
        self.table = table

    def __call__(self, label: str, sid: int, submit: str) -> float:
        return self.table[submit, label]


class UniformWeighter(TableWeighter):
    @classmethod
    def build(cls):
        return cls(table=defaultdict(lambda: 1))


class F1Weighter(TableWeighter):
    @classmethod
    def build(cls, choir: EnsembleChoir, *, entities=ENTITIES, relations=RELATIONS):
        table = {}
        for label in ENTITIES + RELATIONS:
            for name, submit in choir.submissions.items():
                table[name, label] = cls._score(
                    choir,
                    submit,
                    label,
                    skipA=label not in entities,  # I know :P
                    skipB=label not in relations,
                )
        return cls(table=table)

    @classmethod
    def _score(
        cls,
        choir: EnsembleChoir,
        submit: Collection,
        label: str,
        skipA: bool,
        skipB: bool,
    ):
        submit_selection = submit.filter(
            keyphrase=lambda k: (True if skipA else k.label == label),
            relation=lambda r: r.label == label,
        )
        gold_selection = choir.gold.filter(
            keyphrase=lambda k: (True if skipA else k.label == label),
            relation=lambda r: r.label == label,
        )
        score = choir.evaluate_scenario(submit_selection, gold_selection, skipA, skipB)
        return score["f1"]


# =======================================================================
# - Scorer --------------------------------------------------------------
# =======================================================================


class Scorer:
    def __call__(
        self, weights: Dict[str, float], annotation, sid: int, label: str
    ) -> float:
        raise NotImplementedError()


class AverageScorer(Scorer):
    def __call__(
        self, weights: Dict[str, float], annotation, sid: int, label: str
    ) -> float:
        return sum(weights.values()) / len(weights)


class MaxScorer(Scorer):
    def __call__(
        self, weights: Dict[str, float], annotation, sid: int, label: str
    ) -> float:
        return max(weights.values())


class TopScorer(Scorer):
    def __init__(self, k, merge):
        self._k = k
        self._merge = merge

    def __call__(
        self, weights: Dict[str, float], annotation, sid: int, label: str
    ) -> float:
        top = sorted(weights.values(), reverse=True)[: self._k]
        return self._merge(top) if top else 0.0


class AverageTopScorer(TopScorer):
    def __init__(self, k: int, strict: bool):
        merge = lambda top: sum(top) / (k if strict else len(top))
        super().__init__(k, merge)


class AggregateTopScorer(TopScorer):
    def __init__(self, k: int):
        merge = lambda top: sum(top)
        super().__init__(k, merge)


class ExpertScorer(Scorer):
    def __init__(self, weighter, choir: EnsembleChoir):
        super().__init__()
        self._weighter = weighter
        self._choir = choir

    @property
    def weight(self):
        return self._weighter

    def __call__(
        self, weights: Dict[str, float], annotation, sid: int, label: str
    ) -> float:
        best = max(
            self.weight(label, sid, submit) for submit in self._choir.submissions
        )
        return float(any(x for x in weights.values() if x >= best))


class GoldOracleScorer(Scorer):
    def __init__(self, choir: EnsembleChoir):
        super().__init__()
        self._choir = choir

    def __call__(
        self, weights: Dict[str, float], annotation, sid: int, label: str
    ) -> float:

        return float(
            self._everyone_voted(weights)
            or self._found_in_gold_and_at_least_one_voted(annotation, sid, label)
        )

    def _everyone_voted(self, weights):
        return len(weights) == len(self._choir.submissions)

    def _found_in_gold_and_at_least_one_voted(self, annotation, sid: int, label: str):
        gold_sentence = self._choir.gold.sentences[sid]
        gold_annotation = gold_sentence.find_first_match(annotation, label)
        return gold_annotation is not None


# =======================================================================
# - Validator -----------------------------------------------------------
# =======================================================================


class Validator:
    def __call__(self, annotation, sid: int, label: str, score: float) -> Optional[str]:
        raise NotImplementedError()


class NonZeroValidator(Validator):
    def __call__(self, annotation, sid: int, label: str, score: float) -> Optional[str]:
        return label if score else None


class ThresholdValidator(Validator):
    def __init__(self, thresholds: Dict[str, float]):
        super().__init__()
        self._thresholds = thresholds

    def __call__(self, annotation, sid: int, label: str, score: float) -> Optional[str]:
        return label if score >= self._thresholds[label] else None


class ConstantThresholdValidator(ThresholdValidator):
    def __init__(self, threshold=0.5):
        super().__init__(thresholds=defaultdict(lambda: threshold))

