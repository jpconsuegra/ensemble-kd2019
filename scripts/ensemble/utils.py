from collections import defaultdict

from scripts.ensemble import EnsembleChoir
from scripts.utils import Collection


def keep_top_k_submissions(choir: EnsembleChoir, k) -> EnsembleChoir:
    filtered = dict(
        list(
            sorted(
                choir.submissions.items(), key=lambda x: choir.eval(x[1]), reverse=True,
            )
        )[:k]
    )
    return EnsembleChoir(filtered, choir.gold)


def keep_named_submissions(choir: EnsembleChoir, names) -> EnsembleChoir:
    filtered = {
        name: submit for name, submit in choir.submissions.items() if name in names
    }
    return EnsembleChoir(filtered, choir.gold)


def keep_best_per_participant(choir: EnsembleChoir) -> EnsembleChoir:
    group_by = defaultdict(list)
    for name, submit in choir.submissions.items():
        username = name.split("/", maxsplit=1)[0]
        group_by[username].append((name, submit))
    filtered = dict(
        max(items, key=lambda x: choir.eval(x[1])) for items in group_by.values()
    )
    return EnsembleChoir(filtered, choir.gold)


def keep_non_annotated_sentences(choir: EnsembleChoir) -> EnsembleChoir:
    gold = choir.gold.clone()
    submissions = {name: submit.clone() for name, submit in choir.submissions.items()}

    for sid in range(len(choir.gold) - 1, -1, -1):
        sentence = choir.gold.sentences[sid]
        if sentence.annotated:
            del gold.sentences[sid]
            for submit in submissions.values():
                del submit.sentences[sid]

    return EnsembleChoir(submissions, gold)


def keep_annotated_sentences(choir: EnsembleChoir) -> EnsembleChoir:
    gold_annotated_sids = choir.gold_annotated_sid()
    submissions = {
        name: Collection([submit.sentences[sid] for sid in gold_annotated_sids])
        for name, submit in choir.submissions.items()
    }
    return EnsembleChoir(submissions, choir.gold_annotated)


def extract_submissions(collection: Collection, choir: EnsembleChoir) -> EnsembleChoir:
    gold = collection.clone()
    submissions = {}
    not_found = set()
    for name, submit in choir.submissions.items():
        sentences = []
        for sid, s in enumerate(collection.sentences):
            match = submit.find_first_match(s.text)
            try:
                sentences.append(match.clone())
            except AttributeError:
                sentences.append(None)
                not_found.add(sid)
        submissions[name] = Collection(sentences)
    for sid in sorted(not_found, reverse=True):
        del gold.sentences[sid]
        for submit in submissions.values():
            del submit.sentences[sid]
    for submit in submissions.values():
        assert len(submit) == len(gold)
    return EnsembleChoir(submissions, gold)
