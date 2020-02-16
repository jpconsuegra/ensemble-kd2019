import itertools as itt
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .score import compute_metrics, subtaskA, subtaskB
from .utils import ENTITIES, RELATIONS, Collection, Keyphrase, Relation, Sentence

STOP = False


class Ensemble:
    def __init__(self):
        self.submissions = []
        self.gold: Collection = None
        self.collection: Collection = None
        self.keyphrases = {}
        self.relations = {}
        self.table = {}

    def load(self, submits: Path, gold: Path, *, scenario="1-main", best=False):
        self._load_gold(gold, scenario)
        self._load_submissions(submits, scenario, best=best)
        self._filter_submissions()
        self._build_union()
        self._build_table()

    def _load_gold(self, gold: Path, scenario="1-main"):
        number = scenario.split("-")[0]
        gold_input = next(gold.glob(f"scenario{scenario}/*put_scenario{number}.txt"))
        self.gold = Collection().load(gold_input)

    def _load_submissions(self, submits: Path, scenario="1-main", *, best=False):
        for userfolder in submits.iterdir():
            submit = self._load_user_submit(userfolder, scenario, best=best)
            self.submissions.extend(submit)

    def _load_user_submit(self, userfolder: Path, scenario="1-main", *, best=False):
        number = scenario.split("-")[0]
        submissions = []
        for submit in userfolder.iterdir():
            submit_input = next(
                submit.glob(f"scenario{scenario}/*put_scenario{number}.txt")
            )
            collection = Collection().load(submit_input)
            if self._is_invalid(collection):
                continue
            submissions.append(collection)
        if best and submissions:
            return [max(submissions, key=lambda x: self._evaluate(x))]
        return submissions

    def _is_invalid(self, submit: Collection):
        error = []
        if len(submit) != len(self.gold):
            error.append("ERROR! Wrong number of sentences")
        for s, g in zip(submit.sentences, self.gold.sentences):
            if s.text != g.text:
                error.append("ERROR! {0}\nvs\n{1}".format(s, g))
                break
        for e in error:
            print(e)
        if STOP and input("Skip [Y|n]") == "n":
            raise Exception(e)
        return bool(error)

    def _evaluate(self, submit: Collection):
        results = self._evaluate_scenario(submit, self.gold)
        return results["f1"]

    def _evaluate_scenario(
        self, submit: Collection, gold: Collection, skipA=False, skipB=False
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
        for i, submit in enumerate(self.submissions):
            for n, sentence in enumerate(submit.sentences):
                reference = self.collection.sentences[n]
                self._aggregate_keyphrases(sentence, reference, i, n)
                self._aggregate_relations(sentence, reference, i, n)

    def _aggregate_keyphrases(
        self, sentence: Sentence, reference: Sentence, submit: int, sid: int
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
        self, sentence: Sentence, reference: Sentence, submit: int, sid: int
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

    def _gold_annotated_sid(self):
        return set(
            i for i, sentence in enumerate(self.gold.sentences) if sentence.annotated
        )


class BinaryEnsemble(Ensemble):
    def _aggregate_keyphrases(
        self, sentence: Sentence, reference: Sentence, submit: int, sid: int
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
        self, sentence: Sentence, reference: Sentence, submit: int, sid: int
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
    def _filter_submissions():
        self.submissions = list(
            sorted(self.submissions, key=self._evaluate, reverse=True)
        )[:k]

    self._filter_submissions = _filter_submissions
    return self


def F1Builder(self: Ensemble) -> Ensemble:
    def _build_table():
        self.table = {}
        for i, submit in enumerate(self.submissions):
            for label in ENTITIES + RELATIONS:
                self.table[i, label] = _score(
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
        best = max(self.table[i, label] for i in range(len(self.submissions)))
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


class SklearnEnsemble(BinaryEnsemble):
    def __init__(self, split=True):
        super().__init__()
        self.modelA = None
        self.modelB = None
        self.split = split

    def load(self, submits, gold, *, scenario="1-main", best=False):
        super().load(submits, gold, scenario=scenario, best=best)
        self.modelA = self._train(self.keyphrases, ENTITIES)
        self.modelB = self._train(self.relations, RELATIONS)

    def _train(self, annotations, labels):
        model = RandomForestClassifier()
        X_train, X_test, y_train, y_test = self._training_data(annotations, labels)
        model.fit(X_train, y_train)
        print("Training score:", model.score(X_train, y_train))
        print("Testing score:", model.score(X_test, y_test))
        return model

    def _training_data(self, annotations, labels):
        selected_sids = self._gold_annotated_sid()
        X = self._build_features(annotations, labels, selected_sids)
        y = self._build_targets(annotations, selected_sids)
        if self.split:
            return train_test_split(X, y, stratify=y)
        else:
            return X, X, y, y

    def _build_features(self, annotations, labels, selected_sids=None):
        features = []
        for (sid, *_), (ann, votes) in annotations.items():
            if selected_sids is None or sid in selected_sids:
                features.append(self._annotation_features(ann.label, votes, labels))
        return np.asarray(features)

    def _annotation_features(self, label, votes, labels):
        label_features = self._label_to_features(label, labels)
        vote_features = self._vote_to_features(votes)
        return np.concatenate([label_features, vote_features])

    def _vote_to_features(self, votes):
        n_votes = len(self.submissions)
        features = np.zeros(n_votes)
        features[list(votes)] = 1
        return features

    def _label_to_features(self, label, labels):
        index = labels.index(label)
        features = np.zeros(len(labels))
        features[index] = 1
        return features

    def _build_targets(self, annotations, selected_sids=None):
        targets = []
        for (sid, *_), (ann, _) in annotations.items():
            if selected_sids is None or sid in selected_sids:
                gold_sentence = self.gold.sentences[sid]
                gold_annotation = gold_sentence.find_first_match(ann)
                targets.append(int(gold_annotation is not None))
        return np.asarray(targets)

    # # Using only this solves the problem but quite slowly
    # def _score_label(self, annotation, label, votes):
    #     features = self._annotation_features(label, votes)
    #     features = features.reshape(1, -1)
    #     return self.model.predict(features)[0]

    def _do_ensemble(self):
        self._do_prediction(self.modelA, self.keyphrases, ENTITIES)
        self._do_prediction(self.modelB, self.relations, RELATIONS)

    def _do_prediction(self, model, annotations, labels):
        features = self._build_features(annotations, labels)
        predictions = model.predict(features)

        assert len(predictions) == len(annotations)
        for (ann, _), pred in tqdm(
            zip(annotations.values(), predictions), total=len(predictions),
        ):
            if pred < 0.5:
                ann.label = None


if __name__ == "__main__":
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
    e = SklearnEnsemble(split=False)
    ps = Path("./data/submissions/all")
    pg = Path("./data/testing")
    e.load(ps, pg, best=True)
    # e.load(ps, pg, best=False)
    e.make()
    print("==== SCORE ====\n", e._evaluate(e.collection))
