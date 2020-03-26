from pathlib import Path
from statistics import mean, quantiles, stdev, variance
from typing import Literal

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tqdm import tqdm

from scripts.ensemble import (
    EnsembleChoir,
    EnsembledCollection,
    EnsembleOrchestrator,
    Ensembler,
)
from scripts.ensemble.ensemblers import (
    AggregateTopScorer,
    AverageScorer,
    AverageTopScorer,
    ConstantThresholdValidator,
    ExpertScorer,
    F1Weighter,
    GoldOracleScorer,
    MajorityValidator,
    ManualVotingEnsembler,
    MaxScorer,
    NonZeroValidator,
    SumScorer,
    ThresholdValidator,
    UniformWeighter,
)
from scripts.ensemble.features import model_handler_assistant
from scripts.ensemble.learning import (
    IsolatedPredictor,
    ModelTrainer,
    PredictiveEnsembler,
    TrainedPredictor,
)
from scripts.ensemble.optimization import optimize_sampler_fn
from scripts.ensemble.utils import (
    extract_submissions,
    keep_best_per_participant,
    keep_top_k_submissions,
)
from scripts.utils import (
    Collection,
    CollectionHandler,
    CollectionV1Handler,
    CollectionV2Handler,
)


def get_majority_ensembler(choir: EnsembleChoir, binary: bool):
    orchestrator = EnsembleOrchestrator(binary=binary)

    weighter = UniformWeighter.build()
    scorer = SumScorer()
    validator = MajorityValidator(len(choir.submissions))

    ensembler = ManualVotingEnsembler(choir, orchestrator, weighter, scorer, validator)
    return ensembler


def get_ensembler(choir: EnsembleChoir, binary: bool):
    orchestrator = EnsembleOrchestrator(binary=binary)

    weighter = UniformWeighter.build()
    scorer = AverageScorer()
    validator = ConstantThresholdValidator(threshold=0.5)

    ensembler = ManualVotingEnsembler(choir, orchestrator, weighter, scorer, validator)
    return ensembler


def get_ensembler_with_top(choir: EnsembleChoir, k: int, binary: bool):
    choir = keep_top_k_submissions(choir, k)
    return get_ensembler(choir, binary=binary)


def get_f1_ensembler(choir: EnsembleChoir, binary: bool, best: bool, top=None):
    choir = keep_best_per_participant(choir) if best else choir
    choir = keep_top_k_submissions(choir, top) if top is not None else choir

    orchestrator = EnsembleOrchestrator(binary=binary)

    weighter = F1Weighter.build(choir)
    scorer = ExpertScorer(weighter, choir, discrete=True)
    validator = NonZeroValidator()

    ensembler = ManualVotingEnsembler(choir, orchestrator, weighter, scorer, validator)
    return ensembler


def get_gold_ensembler(choir: EnsembleChoir, binary: bool):
    orchestrator = EnsembleOrchestrator(binary=binary)

    weighter = UniformWeighter.build()  # F1Weighter.build(choir)
    scorer = GoldOracleScorer(choir)
    validator = NonZeroValidator()

    ensembler = ManualVotingEnsembler(choir, orchestrator, weighter, scorer, validator)
    return ensembler


def get_trained_predictor(
    reference: EnsembledCollection,
    model_type,
    mode: Literal["category", "all", "each"],
    ignore=(),
):
    handler = model_handler_assistant(
        voters=reference.choir.submissions.keys(),
        model_init=lambda: model_type(random_state=0),
        mode=mode,
    )

    return TrainedPredictor(
        reference, 0.5, trainer=ModelTrainer(handler), ignore=ignore
    )


def get_sklearn_ensembler(
    choir: EnsembleChoir,
    model_type,
    mode: Literal["category", "all", "each"],
    ignore=(),
):
    orchestrator = EnsembleOrchestrator(binary=True)

    reference = orchestrator(choir)
    predictor = get_trained_predictor(reference, model_type, mode, ignore=ignore)

    ensembler = PredictiveEnsembler(choir, orchestrator, predictor)
    return ensembler


def get_isolated_ensembler(
    choir: EnsembleChoir,
    taskA_choir: EnsembleChoir,
    taskB_choir: EnsembleChoir,
    model_type,
    mode: Literal["category", "all", "each"],
    ignore=(),
):
    orchestrator = EnsembleOrchestrator(binary=True)

    taskA_predictor = get_trained_predictor(
        orchestrator(taskA_choir), model_type, mode, ignore=ignore
    )
    taskB_predictor = get_trained_predictor(
        orchestrator(taskB_choir), model_type, mode, ignore=ignore
    )
    predictor = IsolatedPredictor(taskA_predictor, taskB_predictor)

    ensembler = PredictiveEnsembler(choir, orchestrator, predictor)
    return ensembler


def get_multisource_ensembler(
    choir: EnsembleChoir,
    taskA_choir: EnsembleChoir,
    taskB_choir: EnsembleChoir,
    model_type,
    mode: Literal["category", "all", "each"],
):
    taskA_choir = EnsembleChoir.merge(choir, taskA_choir)
    taskB_choir = EnsembleChoir.merge(choir, taskB_choir)
    return get_isolated_ensembler(choir, taskA_choir, taskB_choir, model_type, mode)


def task_run(ensembler: Ensembler, gold=None):
    gold = gold or ensembler.choir.gold
    ensembled = ensembler()
    print("==== SCORE ====\n", EnsembleChoir.evaluate(ensembled, gold, clamp=True))


def validate_model(
    choir: EnsembleChoir,
    taskA_choir: EnsembleChoir,
    taskB_choir: EnsembleChoir,
    model_type,
    mode: Literal["category", "all", "each"],
    limit=None,
):
    taskA_choir = EnsembleChoir.merge(choir, taskA_choir)
    taskB_choir = EnsembleChoir.merge(choir, taskB_choir)

    selected_sids = choir.gold_annotated_sid()
    if limit is not None:
        selected_sids = list(selected_sids)[:limit]

    with open("log.txt", mode="w") as fd:
        scores = []

        for sid in tqdm(selected_sids):
            gold = Collection([choir.gold.sentences[sid]])
            submissions = {
                name: Collection([submit.sentences[sid]])
                for name, submit in choir.submissions.items()
            }
            single_choir = EnsembleChoir(submissions, gold)

            ensembler = get_isolated_ensembler(
                single_choir, taskA_choir, taskB_choir, model_type, mode, ignore=(sid,)
            )
            ensembled = ensembler()
            score = single_choir.eval(ensembled)
            scores.append(score)
            fd.write(f"{sid} -> {score}\n")
            fd.flush()

        return scores


def task_cross_validate(
    choir: EnsembleChoir,
    taskA_choir: EnsembleChoir,
    taskB_choir: EnsembleChoir,
    model_type,
    mode: Literal["category", "all", "each"],
    limit=None,
):
    scores = validate_model(choir, taskA_choir, taskB_choir, model_type, mode, limit)
    print("\n".join(str(x) for x in scores))
    print("=================================")
    print("|= mean ======", mean(scores))
    print("|= stdev =====", stdev(scores))
    print("|= variance ==", variance(scores))
    print("|= quantiles =", quantiles(scores))
    print("=================================")


def task_validate_submission(choir: EnsembleChoir, name: str, gold: Collection):
    choir = keep_best_per_participant(choir)
    submissions = [
        s for sname, s in choir.submissions.items() if sname.split("/")[0] == name
    ]
    assert len(submissions) == 1
    submit = submissions[0]
    print("==== SCORE ====\n", EnsembleChoir.evaluate(submit, gold, clamp=True))


def task_extract(
    outputdir: Path,
    path2ehealth19_submissions: Path,
    path2ehealth19_gold: Path,
    path2ehealth20_ann: Path,
):
    collection = CollectionV2Handler.load_dir(
        Collection(), path2ehealth20_ann, attributes=False
    )
    choir = EnsembleChoir().load(
        CollectionV1Handler,
        path2ehealth19_submissions,
        path2ehealth19_gold,
        scenario="1-main",
        best=False,
    )

    selection = extract_submissions(collection, choir)

    output: Path = outputdir / "testing" / "scenario1-main"
    output.mkdir(parents=True)
    CollectionV1Handler.dump(selection.gold, output / "input_scenario1.txt")

    for name, submit in selection.submissions.items():
        output: Path = outputdir / "submissions" / name / "scenario1-main"
        output.mkdir(parents=True)
        CollectionV1Handler.dump(submit, output / "output_scenario1.txt")


if __name__ == "__main__":
    path2ehealth19_submissions = Path("./data/ehealth2019/submissions/all")
    path2ehealth19_gold = Path("./data/ehealth2019/testing")
    path2scenario1 = path2ehealth19_gold / "scenario1-main"
    path2scenario2 = path2ehealth19_gold / "scenario2-taskA"
    path2scenario3 = path2ehealth19_gold / "scenario3-taskB"
    path2ehealth20_submissions = Path("./data/ehealth2020/submissions")
    path2ehealth20_gold = Path("./data/ehealth2020/testing")
    path2ehealth20_ann = Path("./data/ehealth2020/ann")

    # task_extract(
    #     Path("./data/ehealth2020"),
    #     path2ehealth19_submissions,
    #     path2ehealth19_gold,
    #     path2ehealth20_ann,
    # )

    submissions = path2ehealth19_submissions
    reference = path2ehealth19_gold
    validation = path2ehealth20_ann
    # validation = path2ehealth20_gold

    print(f"======== Loading ... (reference) ==============")
    choir = EnsembleChoir().load(CollectionV1Handler, submissions, reference)
    print("======== Done! =================================")

    print(f"======== Loading ... (validation) ==============")
    validation = CollectionV2Handler.load_dir(
        Collection(), validation, attributes=False
    )
    # validation = CollectionV1Handler.load_dir(
    #     Collection(), validation
    # )
    print("======== Done! =================================")

    # ensembler = get_majority_ensembler(choir, binary=True)
    ensembler = get_f1_ensembler(choir, binary=True, best=True, top=None)
    # ensembler = get_sklearn_ensembler(
    #     choir, model_type=RandomForestClassifier, mode="category"
    # )
    # ensembler = get_isolated_ensembler(
    #     choir,
    #     taskA_choir,
    #     taskB_choir,
    #     model_type=RandomForestClassifier,
    #     mode="category",
    # )
    # ensembler = get_multisource_ensembler(
    #     choir, taskA_choir, taskB_choir, model_type=SVC, mode="category",
    # )
    # ensembler = get_multisource_ensembler(
    #     choir, taskA_choir, taskB_choir, model_type=LogisticRegression, mode="each"
    # )  # 0.6611026808295397

    # task_run(ensembler, choir.gold_annotated)
    # task_run(ensembler, validation)
    # task_validate_submission(choir, 'talp', validation)
    # optimize_sampler_fn(choir, main_choir.gold_annotated, generations=500, pop_size=10)
    # task_cross_validate(
    #     choir,
    #     taskA_choir,
    #     taskB_choir,
    #     model_type=RandomForestClassifier,
    #     mode="category",
    #     limit=None,
    # )
