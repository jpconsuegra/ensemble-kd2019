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

