from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

    def load(self, submits: Path, gold: Path, *, best=False):
        self._load_gold(gold)
        self._load_submissions(submits, best=best)
        self._filter_submissions()
        self._build_union()
        self._build_table()

    def _load_gold(self, gold: Path):
        gold_input = next(gold.glob("scenario1-main/*put_scenario1.txt"))
        self.gold = Collection().load(gold_input)

    def _load_submissions(self, submits: Path, *, best=False):
        for userfolder in submits.iterdir():
            submit = self._load_user_submit(userfolder, best=best)
            self.submissions.extend(submit)

    def _load_user_submit(self, userfolder: Path, *, best=False):
        submissions = []
        for submit in userfolder.iterdir():
            submit_input = next(submit.glob("scenario1-main/*put_scenario1.txt"))
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
            sentence.relations = [r for r in sentence.relations if r.label is not None]

    def _do_ensemble(self):
        for keyphrase, info in self.keyphrases.values():
            self._assign_label(keyphrase, info)
        for relation, info in self.relations.values():
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
        for keyphrase, info in self.keyphrases.values():
            self._assign_label(keyphrase, info)
        for relation, info in self.relations.values():
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
    def __init__(self):
        super().__init__()
        self.model = None

    def load(self, submits, gold, *, best=False):
        super().load(submits, gold, best=best)
        self._train()

    def _train(self):
        model = self.model = RandomForestClassifier()
        X_train, X_test, y_train, y_test = self._training_data()
        model.fit(X_train, y_train)

    def _training_data(self):
        selected_sids = self._gold_annotated_sid()
        X = self._build_features(selected_sids)
        y = self._build_targets(selected_sids)
        return train_test_split(X, y, stratify=y)

    def _build_features(self, selected_sids):
        features = []
        for (sid, *_), (kp, votes) in self.keyphrases.items():
            if sid in selected_sids:
                features.append(self._annotation_features(kp.label, votes))
        for (sid, *_), (rel, votes) in self.relations.items():
            if sid in selected_sids:
                features.append(self._annotation_features(rel.label, votes))
        return np.asarray(features)

    def _annotation_features(self, label, votes):
        label_features = self._label_to_features(label)
        vote_features = self._vote_to_features(votes)
        return np.concatenate([label_features, vote_features])

    def _vote_to_features(self, votes):
        n_votes = len(self.submissions)
        features = np.zeros(n_votes)
        features[list(votes)] = 1
        return features

    def _label_to_features(self, label):
        labels = ENTITIES + RELATIONS
        index = labels.index(label)
        features = np.zeros(len(labels))
        features[index] = 1
        return features

    def _build_targets(self, selected_sids):
        targets = []
        for (sid, *_), (kp, _) in self.keyphrases.items():
            if sid in selected_sids:
                gold_sentence = self.gold.sentences[sid]
                assert gold_sentence.annotated
                gold_annotation = gold_sentence.find_first_match(kp)
                targets.append(int(gold_annotation is not None))
        for (sid, *_), (rel, _) in self.relations.items():
            if sid in selected_sids:
                gold_sentence = self.gold.sentences[sid]
                assert gold_sentence.annotated
                gold_annotation = gold_sentence.find_first_match(rel)
                targets.append(int(gold_annotation is not None))
        return np.asarray(targets)


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
    e = SklearnEnsemble()
    ps = Path("./data/submissions/all")
    pg = Path("./data/testing")
    e.load(ps, pg, best=True)
    # e.load(ps, pg, best=False)
    e.make()
    print("==== SCORE ====\n", e._evaluate(e.collection))
