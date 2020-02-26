import itertools as itt
from collections import defaultdict
from pathlib import Path
from statistics import mean, quantiles, stdev, variance

import numpy as np
from autogoal import optimize
from autogoal.grammar import Continuous
from autogoal.search import ConsoleLogger, ProgressLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tqdm import tqdm

from .score import compute_metrics, subtaskA, subtaskB
from .utils import ENTITIES, RELATIONS, Collection, Keyphrase, Relation, Sentence

STOP = False


class Ensemble:
    def __init__(self):
        self.submissions = {}
        self.gold: Collection = None
        self.collection: Collection = None
        self.keyphrases = {}
        self.relations = {}
        self.table = {}
        self.load_stack = []

    def load(self, submits: Path, gold: Path, *, scenario="1-main", best=False):
        self._load_data(submits, gold, scenario=scenario, best=best)
        self._filter_submissions()

    def build(self):
        self.keyphrases = {}
        self.relations = {}
        self._build_union()
        self._build_table()

    def _load_data(self, submits: Path, gold: Path, *, scenario="1-main", best=False):
        self._load_gold(gold, scenario)
        self._load_submissions(submits, scenario, best=best)

    def _load_gold(self, gold: Path, scenario="1-main"):
        number = scenario.split("-")[0]
        gold_input = next(gold.glob(f"scenario{scenario}/*put_scenario{number}.txt"))
        gold_collection = Collection().load(gold_input)
        self.gold = (
            gold_collection if self.gold is None else self.gold.merge(gold_collection)
        )
        self.load_stack.append(len(gold_collection))

    def _load_submissions(self, submits: Path, scenario="1-main", *, best=False):
        if submits is None:
            print("No submission directory provided!")
            return
        for userfolder in submits.iterdir():
            submit = self._load_user_submit(userfolder, scenario, best=best)
            self._update_submissions(submit)

    def _update_submissions(self, submit):
        for key, value in submit.items():
            try:
                previous_submit = self.submissions[key]
                self.submissions[key] = previous_submit.merge(value)
            except KeyError:
                self.submissions[key] = value

    def _load_user_submit(self, userfolder: Path, scenario="1-main", *, best=False):
        number = scenario.split("-")[0]
        submissions = {}
        for submit in userfolder.iterdir():
            submit_input = next(
                submit.glob(f"scenario{scenario}/*put_scenario{number}.txt")
            )
            collection = Collection().load(submit_input)
            name = f"{userfolder.name}/{submit.name}"
            if self._is_invalid(collection, name):
                continue
            submissions[name] = collection
        if best and submissions:
            return dict([max(submissions.items(), key=lambda x: self.eval(x[1]))])
        return submissions

    def _is_invalid(self, submit: Collection, name: str):
        error = []
        if (len(submit) + self._current_count(name)) != len(self.gold):
            error.append("ERROR! Wrong number of sentences")
        for s, g in zip(submit.sentences, self.gold.sentences[-len(submit) :]):
            if s.text != g.text:
                error.append("ERROR! {0}\nvs\n{1}".format(s, g))
                break
        for e in error:
            print(e)
        if STOP and input("Skip [Y|n]") == "n":
            raise Exception(e)
        return bool(error)

    def _current_count(self, submit_name):
        return len(self.submissions.get(submit_name, []))

    def eval(self, submit: Collection = None):
        return self.evaluate(self.collection if submit is None else submit, self.gold)

    @classmethod
    def evaluate(cls, submit: Collection, gold: Collection, skipA=False, skipB=False):
        results = cls._evaluate_scenario(submit, gold, skipA=skipA, skipB=skipB)
        return results["f1"]

    @staticmethod
    def _evaluate_scenario(
        submit: Collection, gold: Collection, skipA=False, skipB=False
    ):
        resultA = subtaskA(gold, submit)
        resultB = subtaskB(gold, submit, resultA)

        results = {}

        for k, v in list(resultA.items()) + list(resultB.items()):
            results[k] = len(v)

        metrics = compute_metrics(dict(resultA, **resultB), skipA, skipB)
        results.update(metrics)

        return results

    def _filter_submissions(self):
        pass

    def _build_union(self):
        self.collection = Collection([Sentence(s.text) for s in self.gold.sentences])
        for name, submit in self.submissions.items():
            for n, sentence in enumerate(submit.sentences):
                reference = self.collection.sentences[n]
                self._aggregate_keyphrases(sentence, reference, name, n)
                self._aggregate_relations(sentence, reference, name, n)

    def _aggregate_keyphrases(
        self, sentence: Sentence, reference: Sentence, submit: str, sid: int
    ):
        for keyphrase in sentence.keyphrases:
            spans = tuple(keyphrase.spans)
            label = keyphrase.label

            try:
                kf, info = self.keyphrases[sid, spans]
            except KeyError:
                kf, info = self.keyphrases[sid, spans] = (
                    Keyphrase(
                        sentence=reference,
                        label=None,
                        id=len(self.keyphrases),
                        spans=list(spans),
                    ),
                    defaultdict(dict),
                )
                reference.keyphrases.append(kf)

            info[label][submit] = 1

    def _aggregate_relations(
        self, sentence: Sentence, reference: Sentence, submit: str, sid: int
    ):
        for relation in sentence.relations:
            _from = relation.from_phrase
            _to = relation.to_phrase
            fspans = tuple(_from.spans)
            tspans = tuple(_to.spans)
            flabel = _from.label
            tlabel = _to.label
            label = relation.label

            try:
                rel, info = self.relations[sid, fspans, tspans]
            except KeyError:
                rel, info = self.relations[sid, fspans, tspans] = (
                    Relation(
                        sentence=reference,
                        origin=self.keyphrases[sid, fspans][0].id,
                        destination=self.keyphrases[sid, tspans][0].id,
                        label=None,
                    ),
                    defaultdict(lambda: defaultdict(dict)),
                )
                reference.relations.append(rel)

            info[flabel, tlabel][label][submit] = 1

    def _build_table(self):
        self.table = defaultdict(lambda: 1)
        self.table[None] = 0.5

    def make(self):
        self._do_ensemble()

        for sentence in self.collection.sentences:
            sentence.keyphrases = [
                s for s in sentence.keyphrases if s.label is not None
            ]
            sentence.relations = [
                r
                for r in sentence.relations
                if r.label is not None
                and r.from_phrase is not None
                and r.from_phrase.label is not None  # don't needed!
                and r.to_phrase is not None
                and r.to_phrase.label is not None  # don't needed!
            ]

    def _do_ensemble(self):
        for keyphrase, info in tqdm(self.keyphrases.values()):
            self._assign_label(keyphrase, info)
        for relation, info in tqdm(self.relations.values()):
            info = info[relation.from_phrase.label, relation.to_phrase.label]
            self._assign_label(relation, info)

    def _assign_label(self, annotation, info: dict):
        metrics = self._compute_metrics(annotation, info)
        annotation.label = self._select_label(annotation, metrics)

    def _compute_metrics(self, annotation, info: dict):
        metrics = {}
        for label, votes in info.items():
            metrics[label] = self._score_label(annotation, label, votes)
        return metrics

    def _score_label(self, annotation, label, votes):
        return sum(self.table[submit, label] for submit in votes) / len(
            self.submissions
        )

    def _select_label(self, annotation, metrics: dict):
        if not metrics:
            return None
        label = max(metrics, key=lambda x: metrics[x])
        return self._validate_label(annotation, label, metrics[label])

    def _validate_label(self, annotation, label, score):
        return label if score > self.table[None] else None

    def gold_annotated_sid(self):
        return set(
            i for i, sentence in enumerate(self.gold.sentences) if sentence.annotated
        )


class BinaryEnsemble(Ensemble):
    def _aggregate_keyphrases(
        self, sentence: Sentence, reference: Sentence, submit: str, sid: int
    ):
        for keyphrase in sentence.keyphrases:
            spans = tuple(keyphrase.spans)
            label = keyphrase.label

            try:
                kf, info = self.keyphrases[sid, spans, label]
            except KeyError:
                kf, info = self.keyphrases[sid, spans, label] = (
                    Keyphrase(
                        sentence=reference,
                        label=label,
                        id=len(self.keyphrases),
                        spans=list(spans),
                    ),
                    dict(),
                )
                reference.keyphrases.append(kf)

            info[submit] = 1

    def _aggregate_relations(
        self, sentence: Sentence, reference: Sentence, submit: str, sid: int
    ):
        for relation in sentence.relations:
            _from = relation.from_phrase
            _to = relation.to_phrase
            fspans = tuple(_from.spans)
            tspans = tuple(_to.spans)
            flabel = _from.label
            tlabel = _to.label
            label = relation.label

            try:
                rel, info = self.relations[sid, fspans, tspans, flabel, tlabel, label]
            except KeyError:
                rel, info = self.relations[
                    sid, fspans, tspans, flabel, tlabel, label
                ] = (
                    Relation(
                        sentence=reference,
                        origin=self.keyphrases[sid, fspans, flabel][0].id,
                        destination=self.keyphrases[sid, tspans, tlabel][0].id,
                        label=label,
                    ),
                    dict(),
                )
                reference.relations.append(rel)

            info[submit] = 1

    def _do_ensemble(self):
        for keyphrase, info in tqdm(self.keyphrases.values()):
            self._assign_label(keyphrase, info)
        for relation, info in tqdm(self.relations.values()):
            if (
                relation.from_phrase.label is not None
                and relation.to_phrase.label is not None
            ):
                self._assign_label(relation, info)
            else:
                relation.label = None

    def _compute_metrics(self, annotation, info: dict):
        return {annotation.label: self._score_label(annotation, annotation.label, info)}


def Top(self: Ensemble, k) -> Ensemble:
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


def F1Builder(self: Ensemble) -> Ensemble:
    def _build_table():
        self.table = {}
        for name, submit in self.submissions.items():
            for label in ENTITIES + RELATIONS:
                self.table[name, label] = _score(
                    submit,
                    label,
                    skipA=label not in ENTITIES,  # I know :P
                    skipB=label not in RELATIONS,
                )
        self.table[None] = 0.5

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


def MaxSelector(self: Ensemble) -> Ensemble:
    def _score_label(annotation, label, votes):
        return max(self.table[submit, label] for submit in votes)

    self._score_label = _score_label
    return self


def BooleanThreadshold(self: Ensemble) -> Ensemble:
    def _validate_label(annotation, label, score):
        return label if score else None

    self._validate_label = _validate_label
    return self


def BestSelector(self: Ensemble) -> Ensemble:
    self = BooleanThreadshold(self)

    def _score_label(annotation, label, votes):
        best = max(self.table[submit, label] for submit in self.submissions)
        return max(self.table[submit, label] >= best for submit in votes)

    self._score_label = _score_label
    return self


def GoldSelector(self: Ensemble) -> Ensemble:
    self = BooleanThreadshold(self)

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


class VotingFeatures:
    def __init__(self, voters):
        self._voters = voters

    def __call__(self, votes):
        n_votes = len(self._voters)
        features = np.zeros(n_votes)
        voted = [i for i, submit in enumerate(self._voters) if submit in votes]
        features[voted] = 1
        return features


class LabelFeatures:
    def __init__(self, labels):
        self._labels = labels

    def __call__(self, label):
        index = self._labels.index(label)
        features = np.zeros(len(self._labels))
        features[index] = 1
        return features


class ConcatenatedFeatures:
    def __init__(self, *builders_and_handlers):
        self._builders_and_handlers = builders_and_handlers

    def __call__(self, item):
        return np.concatenate(
            [builder(handler(item)) for builder, handler in self._builders_and_handlers]
        )


class AllInOneModel:
    def __init__(self, *, voters, labels, model_init):
        self._builder = ConcatenatedFeatures(
            (LabelFeatures(labels), self._get_label),
            (VotingFeatures(voters), self._get_votes),
        )
        self._model = model_init()

    def _get_label(self, item):
        ann, _ = item
        return ann.label

    def _get_votes(self, item):
        _, votes = item
        return votes

    def __call__(self, annotations, selected_sids=None):
        sids = []
        anns = []
        features = []
        for (sid, *_), (ann, votes) in annotations.items():
            if selected_sids is None or sid in selected_sids:
                features.append(self._builder((ann, votes)))
                sids.append(sid)
                anns.append(ann)
        return [("all", self._model, sids, anns, np.asarray(features))]


class PerLabelModel:
    def __init__(self, *, voters, labels, model_init):
        self._builder = VotingFeatures(voters)
        self._models = {label: model_init() for label in labels}

    def __call__(self, annotations, selected_sids=None):
        per_label = defaultdict(lambda: ([], [], []))
        for (sid, *_), (ann, votes) in annotations.items():
            if selected_sids is None or sid in selected_sids:
                sids, anns, features = per_label[ann.label]

                features.append(self._builder(votes))
                sids.append(sid)
                anns.append(ann)

        return [
            (label, self._models[label], sids, anns, np.asarray(features))
            for label, (sids, anns, features) in per_label.items()
        ]


class PredictiveEnsemble(BinaryEnsemble):
    def _do_ensemble(self):
        raise NotImplementedError()

    def _do_prediction(self, model_handler, annotations, labels):
        for _, model, _, anns, features in model_handler(annotations):
            predictions = model.predict(features)

            assert len(predictions) == len(anns)
            for ann, pred in tqdm(zip(anns, predictions), total=len(anns)):
                if pred < 0.5:
                    ann.label = None


class TrainableEnsemble(PredictiveEnsemble):
    def __init__(self, model_handler_init=AllInOneModel):
        super().__init__()
        self._model_handler_init = model_handler_init

    def _train(self, annotations, labels, ignore=()):
        selected_sids = {x for x in self.gold_annotated_sid() if x not in ignore}

        model_handler = self._model_handler_init(
            voters=self.submissions, labels=labels, model_init=self._init_model
        )

        for tag, model, sids, anns, X in model_handler(
            annotations, selected_sids=selected_sids
        ):
            y = self._build_targets(anns, sids)
            model.fit(X, y)
            print(f"Training score ({tag}):", model.score(X, y))

        return model_handler

    def _init_model(self):
        raise NotImplementedError()

    def _build_targets(self, annotations, sids):
        targets = []
        for ann, sid in zip(annotations, sids):
            gold_sentence = self.gold.sentences[sid]
            gold_annotation = gold_sentence.find_first_match(ann)
            targets.append(int(gold_annotation is not None))
        return np.asarray(targets)


class SklearnEnsemble(TrainableEnsemble):
    def __init__(
        self, model_type=RandomForestClassifier, model_handler_init=AllInOneModel
    ):
        super().__init__(model_handler_init=model_handler_init)
        self._model_type = model_type
        self.modelA = None
        self.modelB = None

    def build(self, ignore=()):
        super().build()
        self.modelA = self._train(self.keyphrases, ENTITIES, ignore)
        self.modelB = self._train(self.relations, RELATIONS, ignore)

    def _init_model(self):
        return self._model_type(random_state=0)

    def _do_ensemble(self):
        self._do_prediction(self.modelA, self.keyphrases, ENTITIES)
        self._do_prediction(self.modelB, self.relations, RELATIONS)


class IsolatedDualEnsemble(PredictiveEnsemble):
    def __init__(
        self, model_type=RandomForestClassifier, model_handler_init=AllInOneModel
    ):
        super().__init__()
        self._model_type = model_type
        self._model_handler_init = model_handler_init
        self.keyphrase_ensemble = None
        self.relation_ensemble = None

    def load(self, submits: Path, gold: Path, *, scenario="1-main", best=False):
        super().load(submits, gold, scenario=scenario, best=best)

        self.keyphrase_ensemble = SklearnEnsemble(
            model_type=self._model_type, model_handler_init=self._model_handler_init
        )
        self.keyphrase_ensemble.load(submits, gold, scenario="2-taskA", best=best)
        self.keyphrase_ensemble.build()

        self.relation_ensemble = SklearnEnsemble(
            model_type=self._model_type, model_handler_init=self._model_handler_init
        )
        self.relation_ensemble.load(submits, gold, scenario="3-taskB", best=best)
        self.relation_ensemble.build()

    def _do_ensemble(self):
        print("Recolected votes", len(self.submissions))
        print("Informative votes (A):", len(self.keyphrase_ensemble.submissions))
        print(
            "Votes (A) in use:",
            len(self.submissions.keys() & self.keyphrase_ensemble.submissions.keys()),
        )
        print("Informative votes (B):", len(self.relation_ensemble.submissions))
        print(
            "Votes (B) in use:",
            len(self.submissions.keys() & self.relation_ensemble.submissions.keys()),
        )
        self.keyphrase_ensemble._do_prediction(
            self.keyphrase_ensemble.modelA, self.keyphrases, ENTITIES
        )
        self.relation_ensemble._do_prediction(
            self.relation_ensemble.modelB, self.relations, RELATIONS
        )


class MultiScenarioSKEmsemble(SklearnEnsemble):
    def _load_data(self, submits, gold, *, scenario="1-main", best=False):
        super()._load_data(submits, gold, scenario="1-main", best=best)
        super()._load_data(submits, gold, scenario="2-taskA", best=best)
        super()._load_data(submits, gold, scenario="3-taskB", best=best)

    def _filter_submissions(self):
        # unnecessary due to loading verification (`is_invalid`)
        super()._filter_submissions()
        old = set(self.submissions)
        print("Gold sentences:", len(self.gold))
        print("Submissions' sentences", [len(x) for x in self.submissions.values()])
        self.submissions = {
            name: collection
            for name, collection in self.submissions.items()
            if len(collection) == len(self.gold)
        }
        removed = old - self.submissions.keys()
        print(f"Removed ({len(removed)} out of {len(old)}):\n", removed)


class MultiSourceEnsemble(PredictiveEnsemble):
    def __init__(
        self, model_type=RandomForestClassifier, model_handler_init=AllInOneModel
    ):
        super().__init__()
        self.ensembler = None
        self._model_type = model_type
        self._model_handler_init = model_handler_init

    def load(self, submits: Path, gold: Path, *, scenario="1-main", best=False):
        super().load(submits, gold, scenario=scenario, best=best)
        self.ensembler = MultiScenarioSKEmsemble(
            model_type=self._model_type, model_handler_init=self._model_handler_init
        )
        self.ensembler.load(submits, gold, best=best)

    def build(self, ignore=()):
        super().build()
        self.ensembler.build(ignore)

    def _do_ensemble(self):
        print("Recolected votes:", len(self.submissions))
        print("Informative votes:", len(self.ensembler.submissions))
        print(
            "Votes in use:",
            len(self.submissions.keys() & self.ensembler.submissions.keys()),
        )
        self.ensembler._do_prediction(self.ensembler.modelA, self.keyphrases, ENTITIES)
        self.ensembler._do_prediction(self.ensembler.modelB, self.relations, RELATIONS)


def validate_model(
    submits: Path,
    gold: Path,
    *,
    best=True,
    limit=None,
    model_type=RandomForestClassifier,
    model_handler_init=AllInOneModel,
):

    ensemble = MultiSourceEnsemble(
        model_type=model_type, model_handler_init=model_handler_init
    )
    ensemble.load(submits, gold, best=best)

    selected_sids = [
        x for x in ensemble.gold_annotated_sid() if x < ensemble.load_stack[0]
    ]
    if limit is not None:
        selected_sids = selected_sids[:limit]

    with open("log.txt", mode="w") as fd:
        scores = []

        for sid in tqdm(selected_sids):
            ensemble.build(ignore=(sid,))
            ensemble.make()
            ensembled_collection = Collection([ensemble.collection.sentences[sid]])
            gold_collection = Collection([ensemble.gold.sentences[sid]])
            score = ensemble.evaluate(ensembled_collection, gold_collection)
            scores.append(score)
            fd.write(f"{sid} -> {score}\n")
            fd.flush()

        return scores


def ExploratoryEnsemble(self: Ensemble, threadshold) -> Ensemble:
    self = F1Builder(self)
    super_build = self._build_table

    def _build_table():
        super_build()
        self.table[None] = threadshold

    self._build_table = _build_table
    return self


def build_fn(ensemble: Ensemble):
    def fn(threadshold: Continuous(0, 1)):
        e = ExploratoryEnsemble(ensemble, threadshold)
        e.build()
        e.make()
        return e.eval()

    return fn


if __name__ == "__main__":

    ps = Path("./data/submissions/all")
    pg = Path("./data/testing")

    # e = Ensemble()
    # e = BinaryEnsemble()
    # e = Top(Ensemble(), 1)
    # e = Top(BinaryEnsemble(), 1)
    # e = F1Builder(BinaryEnsemble())
    # e = MaxSelector(F1Builder(Ensemble()))
    # e = BestSelector(F1Builder(Ensemble()))
    # e = BestSelector(F1Builder(BinaryEnsemble()))
    # e = GoldSelector(F1Builder(Ensemble()))
    # e = GoldSelector(F1Builder(BinaryEnsemble()))
    # e = SklearnEnsemble()
    # e = SklearnEnsemble(model_handler_init=PerLabelModel)
    # e = IsolatedDualEnsemble()
    # e = IsolatedDualEnsemble(model_handler_init=PerLabelModel)
    # e = MultiScenarioSKEmsemble()
    # e = MultiScenarioSKEmsemble(model_handler_init=PerLabelModel)
    # e = MultiSourceEnsemble()
    # e = MultiSourceEnsemble(model_type=SVC, model_handler_init=PerLabelModel)
    # e = ExploratoryEnsemble(BinaryEnsemble(), 0.5)  # 0.5 ~ F1Builder(BinaryEnsemble())

    # e.load(ps, pg, best=True)
    # e.load(ps, pg, best=False)

    # e.build()
    # e.make()
    # print("==== SCORE ====\n", e.eval())

    scores = validate_model(ps, pg, best=False, model_type=RandomForestClassifier)
    print("\n".join(str(x) for x in scores))
    print("=================================")
    print("|= mean ======", mean(scores))
    print("|= stdev =====", stdev(scores))
    print("|= variance ==", variance(scores))
    print("|= quantiles =", quantiles(scores))
    print("=================================")

    # loggers = [ProgressLogger(), ConsoleLogger()]
    # print(optimize(build_fn(e), logger=loggers, iterations=5))
