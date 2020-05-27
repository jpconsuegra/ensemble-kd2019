from pathlib import Path

import fire

from scripts.ensemble import EnsembleChoir
from scripts.ensemble.optimization import get_custom_ensembler, optimize_sampler_fn
from scripts.ensemble.utils import exclude_from, keep_non_annotated_sentences
from scripts.main import task_do_ensemble
from scripts.utils import CollectionV1Handler, CollectionV2Handler


def do(
    ref_submissions: Path,
    ref_gold: Path,
    ref_version: int,
    ref_scenario: str,
    ref_cname: str,
    target_submissions: Path,
    target_gold: Path,
    target_version: int,
    target_scenario: str,
    target_cname: str,
    output_text: Path,
    generations=500,
    pop_size=10,
    manual_voting=True,
    learning=True,
    configuration=None,
    exclude_gold_annotated: bool = False,
    top: int = None,
):
    (
        ref_submissions,
        ref_gold,
        target_submissions,
        target_gold,
        output_text,
    ) = turn_into_path(
        ref_submissions, ref_gold, target_submissions, target_gold, output_text
    )

    handler1 = get_handler(ref_version)
    print(f" Loading ... (reference) ".center(48, "="))
    choir = EnsembleChoir().load(
        handler1, ref_submissions, ref_gold, scenario=ref_scenario, cname=ref_cname
    )
    print(" Done! ".center(48, "="))

    print(f"Working on ({len(choir.submissions)}):")
    print("\n".join(choir.submissions.keys()))

    handler2 = get_handler(target_version)
    print(f" Loading ... (target) ".center(48, "="))
    target = EnsembleChoir().load(
        handler2,
        target_submissions,
        target_gold,
        scenario=target_scenario,
        cname=target_cname,
    )
    print(" Done! ".center(48, "="))

    print(f"Working on ({len(target.submissions)}):")
    print("\n".join(target.submissions.keys()))

    if configuration is None:
        print(" Optimizing ".center(48, "="))
        best, best_fd = optimize_sampler_fn(
            choir,
            choir.gold_annotated,
            generations=generations,
            pop_size=pop_size,
            manual_voting=manual_voting,
            learning=learning,
        )
        print(f" Done! With F1: {best_fd} ".center(48, "="))

        print(" Ensembling target collection ".center(48, "="))
        ensembler = best.model

    else:
        ensembler = get_custom_ensembler(choir, target, configuration)

    if exclude_gold_annotated:
        print(len(target.gold), len(target.gold_annotated))
        target = keep_non_annotated_sentences(target)
        print(len(target.gold))
        print(len(choir.gold), len(choir.gold_annotated))
        target = exclude_from(target, choir.gold_annotated)
        print(len(target.gold))

    print(f" Ensembling to {output_text} ".center(48, "="))
    task_do_ensemble(ensembler, target, top=top, opath=output_text)
    print(" Done! ".center(48, "="))


def turn_into_path(*path):
    return tuple(Path(s) for s in path if not isinstance(path, Path))


def get_handler(version):
    if version == 1:
        return CollectionV1Handler
    if version == 2:
        return CollectionV2Handler
    raise ValueError(f"Unknown handler version: {version}!!!")


if __name__ == "__main__":
    fire.Fire(do)
