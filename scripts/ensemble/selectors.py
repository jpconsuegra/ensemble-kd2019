from scripts.ensemble.models import Ensemble
from scripts.utils import ENTITIES, RELATIONS, Collection, Keyphrase, Relation, Sentence


def KeepTopKSubmissions(self: Ensemble, k) -> Ensemble:
    super_filter = self._filter_submissions

    def _filter_submissions():
        super_filter()
        self.submissions = dict(
            list(
                sorted(
                    self.submissions.items(),
                    key=lambda x: self.eval(x[1]),
                    reverse=True,
                )
            )[:k]
        )

    self._filter_submissions = _filter_submissions
    return self


def WeightVotesWithF1Score(self: Ensemble) -> Ensemble:
    def _build_table():
        self.table = {}
        for label in ENTITIES + RELATIONS:
            for name, submit in self.submissions.items():
                self.table[name, label] = _score(
                    submit,
                    label,
                    skipA=label not in ENTITIES,  # I know :P
                    skipB=label not in RELATIONS,
                )
            self.table[label] = 0.5

    def _score(submit: Collection, label: str, skipA, skipB):
        submit_selection = submit.filter(
            keyphrase=lambda k: (True if skipA else k.label == label),
            relation=lambda r: r.label == label,
        )
        gold_selection = self.gold.filter(
            keyphrase=lambda k: (True if skipA else k.label == label),
            relation=lambda r: r.label == label,
        )
        score = self._evaluate_scenario(submit_selection, gold_selection, skipA, skipB)
        return score["f1"]

    self._build_table = _build_table
    return self


def ScoreLabelAsMaxVoteWeight(self: Ensemble) -> Ensemble:
    def _score_label(annotation, label, votes):
        return max(self.table[submit, label] for submit in votes)

    self._score_label = _score_label
    return self


def AcceptLabelWithNonZeroScore(self: Ensemble) -> Ensemble:
    def _validate_label(annotation, label, score):
        return label if score else None

    self._validate_label = _validate_label
    return self


def AcceptLabelIfExpertVoted(self: Ensemble) -> Ensemble:
    self = AcceptLabelWithNonZeroScore(self)

    def _score_label(annotation, label, votes):
        best = max(self.table[submit, label] for submit in self.submissions)
        return max(self.table[submit, label] >= best for submit in votes)

    self._score_label = _score_label
    return self


def AcceptLabelWithGoldOracle(self: Ensemble) -> Ensemble:
    self = AcceptLabelWithNonZeroScore(self)

    def _score_label(annotation, label, votes):
        score = False

        for sentence, gold_sentence in zip(
            self.collection.sentences, self.gold.sentences
        ):
            if sentence == annotation.sentence:
                break
        else:
            raise Exception("Not found")

        gold_annotation = gold_sentence.find_first_match(annotation, label)
        score |= gold_annotation is not None

        score |= len(votes) == len(self.submissions)

        return score

    self._score_label = _score_label
    return self


def SetParameterAsThreshold(self: Ensemble, thresholds) -> Ensemble:
    super_build = self._build_table

    def _build_table():
        super_build()
        for label, threshold in thresholds:
            self.table[label] = threshold

    self._build_table = _build_table
    return self
