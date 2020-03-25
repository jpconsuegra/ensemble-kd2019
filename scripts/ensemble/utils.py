from collections import defaultdict

from scripts.ensemble import EnsembleChoir


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
