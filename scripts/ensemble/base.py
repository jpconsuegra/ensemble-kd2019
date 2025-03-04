import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

from scripts.score import compute_macro_f1, compute_metrics, subtaskA, subtaskB
from scripts.utils import Collection, CollectionHandler, Keyphrase, Relation, Sentence

STOP = False


class EnsembleChoir:
    def __init__(
        self, submissions=None, gold: Collection = None,
    ):
        self.submissions = submissions or {}
        self.gold: Collection = gold

    @property
    def sentences(self):
        return self.gold.sentences

    @property
    def gold_annotated(self):
        return Collection([s for s in self.gold.sentences if s.annotated])

    def load(
        self,
        handler: CollectionHandler,
        submits: Path,
        gold: Path,
        *,
        scenario="1-main",
        cname="",
        best=False,
    ):
        self._load_data(
            handler, submits, gold, scenario=scenario, cname=cname, best=best
        )
        return self

    def _load_data(
        self,
        handler: CollectionHandler,
        submits: Path,
        gold: Path,
        *,
        scenario="1-main",
        cname="",
        best=False,
    ):
        self._load_gold(handler, gold, scenario)
        self._load_submissions(handler, submits, scenario, cname, best=best)

    def _load_gold(self, handler: CollectionHandler, gold: Path, scenario="1-main"):
        gold_collection = handler.load_dir(Collection(), gold / f"scenario{scenario}")
        self._update_gold(gold_collection)

    def _update_gold(self, gold: Collection):
        self.gold = gold if self.gold is None else self.gold.merge(gold)

    def _load_submissions(
        self,
        handler: CollectionHandler,
        submits: Path,
        scenario="1-main",
        cname="",
        *,
        best=False,
    ):
        if submits is None:
            print("No submission directory provided!")
            return
        for userfolder in submits.iterdir():
            if not userfolder.is_dir():
                continue
            submit = self._load_user_submit(
                handler, userfolder, scenario, cname, best=best
            )
            self._update_submissions(submit)

    def _update_submissions(self, submit, strict=False):
        for key, value in submit.items():
            try:
                previous_submit = self.submissions[key]
                self.submissions[key] = previous_submit.merge(value)
            except KeyError:
                if strict:
                    continue
                self.submissions[key] = value

    def _load_user_submit(
        self,
        handler: CollectionHandler,
        userfolder: Path,
        scenario="1-main",
        cname="",
        *,
        best=False,
    ):
        submissions = {}
        for submit in (userfolder / cname).iterdir():
            collection = handler.load_dir(Collection(), submit / f"scenario{scenario}")
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
            error.append(f"ERROR! Wrong number of sentences at {name}")
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

    def eval(
        self,
        submit: Collection,
        *,
        skipA=False,
        skipB=False,
        clamp=False,
        macro=False,
        all_metrics=False,
    ):
        return self.evaluate(
            submit,
            self.gold,
            skipA=skipA,
            skipB=skipB,
            clamp=clamp,
            macro=macro,
            all_metrics=all_metrics,
        )

    @classmethod
    def evaluate(
        cls,
        submit: Collection,
        gold: Collection,
        skipA=False,
        skipB=False,
        clamp=False,
        macro=False,
        all_metrics=False,
    ):
        if clamp:
            sentences = []
            submit = submit.clone()
            for gold_sent in gold.sentences:
                matches = submit.find_matches(gold_sent.text, get_id=True)
                if not matches:
                    warnings.warn("Sentence not found!")
                    sentences.append(Sentence(gold_sent.text))
                elif len(matches) > 1:
                    warnings.warn("Ambiguous collection")
                sentence, sid = matches[0]
                sentences.append(sentence)
                del submit.sentences[sid]

            submit = Collection(sentences)

        results = cls.evaluate_scenario(
            submit, gold, skipA=skipA, skipB=skipB, macro=macro
        )
        return (
            results if all_metrics else results["f1"] if not macro else results["macro"]
        )

    @staticmethod
    def evaluate_scenario(
        submit: Collection, gold: Collection, skipA=False, skipB=False, macro=False
    ):
        resultA = subtaskA(gold, submit)
        resultB = subtaskB(gold, submit, resultA)

        results = {}

        for k, v in list(resultA.items()) + list(resultB.items()):
            results[k] = len(v)

        data = dict(resultA, **resultB)

        metrics = compute_metrics(data, skipA, skipB)
        results.update(metrics)

        if macro:
            results.update(compute_macro_f1(data, skipA, skipB))

        return results

    def gold_annotated_sid(self):
        return set(
            i for i, sentence in enumerate(self.gold.sentences) if sentence.annotated
        )

    @staticmethod
    def merge(*choirs: "EnsembleChoir"):
        choir = EnsembleChoir()
        strict = False
        for other in choirs:
            choir._update_gold(other.gold)
            choir._update_submissions(other.submissions, strict=strict)
            choir.clamp()
            strict = True
        return choir

    def clamp(self):
        keys = list(self.submissions)
        for k in keys:
            if len(self.submissions[k]) != len(self.gold):
                del self.submissions[k]


class EnsembledCollection:
    def __init__(
        self, choir: EnsembleChoir, collection: Collection, keyphrases, relations
    ):
        self._choir = choir
        self._collection = collection
        self._keyphrases = keyphrases
        self._relations = relations

    @property
    def collection(self):
        return self._collection

    @property
    def sentences(self):
        return self._collection.sentences

    @property
    def choir(self):
        return self._choir

    def eval(self, macro=False):
        return self.choir.eval(self._collection, macro=macro)

    @classmethod
    def build(cls, choir):
        collection = Collection([Sentence(s.text) for s in choir.sentences])
        keyphrases, relations = cls._build_votes(choir, collection)
        return cls(choir, collection, keyphrases, relations)

    @classmethod
    def _build_votes(cls, choir, collection: Collection):
        keyphrases = {}
        relations = {}

        for name, submit in choir.submissions.items():
            for sid, sentence in enumerate(submit.sentences):
                reference = collection.sentences[sid]
                cls._fill_keyphrase_votes(keyphrases, sentence, reference, name, sid)
                cls._fill_relation_votes(
                    keyphrases, relations, sentence, reference, name, sid
                )

        return keyphrases, relations

    @classmethod
    def _fill_keyphrase_votes(
        cls,
        keyphrases: dict,
        sentence: Sentence,
        reference: Sentence,
        name: str,
        sid: int,
    ):
        for keyphrase in sentence.keyphrases:
            spans = tuple(keyphrase.spans)
            label = keyphrase.label
            try:
                kf, info = keyphrases[sid, spans]
            except KeyError:
                kf, info = keyphrases[sid, spans] = (
                    Keyphrase(
                        sentence=reference,
                        label=None,
                        id=len(keyphrases),
                        spans=list(spans),
                    ),
                    defaultdict(dict),
                )
                reference.keyphrases.append(kf)
            info[label][name] = 1

    @classmethod
    def _fill_relation_votes(
        cls,
        keyphrases: dict,
        relations: dict,
        sentence: Sentence,
        reference: Sentence,
        name: str,
        sid: int,
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
                rel, info = relations[sid, fspans, tspans, flabel, tlabel]
            except KeyError:
                rel, info = relations[sid, fspans, tspans, flabel, tlabel] = (
                    Relation(
                        sentence=reference,
                        origin=keyphrases[sid, fspans][0].id,
                        destination=keyphrases[sid, tspans][0].id,
                        label=None,
                    ),
                    defaultdict(dict),
                )
                reference.relations.append(rel)

            info[label][name] = 1

    def keyphrase_votes(self):
        return list(self._keyphrase_votes())

    def _keyphrase_votes(self):
        for (sid, *_), (ann, votes_per_label) in self._keyphrases.items():
            yield sid, ann, votes_per_label

    def relation_votes(self, only_valid=True):
        return list(self._relation_votes(only_valid))

    def _relation_votes(self, only_valid=True):
        for (
            (sid, *_, flabel, tlabel),
            (ann, votes_per_label),
        ) in self._relations.items():
            try:
                if not only_valid or (
                    ann.from_phrase.label == flabel and ann.to_phrase.label == tlabel
                ):
                    yield sid, ann, votes_per_label
                else:
                    raise AttributeError
            except AttributeError:
                yield sid, ann, {}


class BinaryEnsembledCollection(EnsembledCollection):
    @classmethod
    def _fill_keyphrase_votes(
        cls,
        keyphrases: dict,
        sentence: Sentence,
        reference: Sentence,
        name: str,
        sid: int,
    ):
        for keyphrase in sentence.keyphrases:
            spans = tuple(keyphrase.spans)
            label = keyphrase.label

            try:
                kf, info = keyphrases[sid, spans, label]
            except KeyError:
                kf, info = keyphrases[sid, spans, label] = (
                    Keyphrase(
                        sentence=reference,
                        label=label,
                        id=len(keyphrases),
                        spans=list(spans),
                    ),
                    dict(),
                )
                reference.keyphrases.append(kf)

            info[name] = 1

    @classmethod
    def _fill_relation_votes(
        cls,
        keyphrases: dict,
        relations: dict,
        sentence: Sentence,
        reference: Sentence,
        name: str,
        sid: int,
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
                rel, info = relations[sid, fspans, tspans, flabel, tlabel, label]
            except KeyError:
                rel, info = relations[sid, fspans, tspans, flabel, tlabel, label] = (
                    Relation(
                        sentence=reference,
                        origin=keyphrases[sid, fspans, flabel][0].id,
                        destination=keyphrases[sid, tspans, tlabel][0].id,
                        label=label,
                    ),
                    dict(),
                )
                reference.relations.append(rel)

            info[name] = 1

    def _keyphrase_votes(self):
        for (sid, *_, label), (ann, votes) in self._keyphrases.items():
            # assert label == ann.label # true first but false after ensembling
            yield sid, ann, {label: votes}

    def _relation_votes(self, only_valid=True):
        for (sid, *_, flabel, tlabel, label), (ann, votes) in self._relations.items():
            try:
                if not only_valid or (
                    ann.from_phrase.label == flabel and ann.to_phrase.label == tlabel
                ):
                    yield sid, ann, {label: votes}
                else:
                    raise AttributeError
            except AttributeError:
                yield sid, ann, {}


class EnsembleOrchestrator:
    def __init__(self, binary):
        self.initializer = BinaryEnsembledCollection if binary else EnsembledCollection

    def __call__(self, choir) -> EnsembledCollection:
        return self.initializer.build(choir)


class Ensembler:
    def __init__(self, choir: EnsembleChoir, orchestrator: EnsembleOrchestrator):
        self._choir = choir
        self._orchestrator = orchestrator

    @property
    def choir(self):
        return self._choir

    def __call__(
        self, choir: EnsembleChoir = None, collection_only=True
    ) -> Union[Collection, EnsembledCollection]:
        choir = choir if choir is not None else self._choir
        to_ensemble = self._orchestrator(choir)
        ensembled = self._ensemble(to_ensemble)
        return ensembled if collection_only else to_ensemble

    def _ensemble(self, to_ensemble: EnsembledCollection) -> Collection:
        self._do_ensemble(to_ensemble)

        for sentence in to_ensemble.sentences:
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

        return to_ensemble.collection

    def _do_ensemble(self, to_ensemble: EnsembledCollection):
        raise NotImplementedError()
