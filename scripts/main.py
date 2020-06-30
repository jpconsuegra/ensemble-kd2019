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
    FocusedScorer,
    GoldOracleScorer,
    MajorityValidator,
    ManualVotingEnsembler,
    MaxScorer,
    NonZeroValidator,
    SumScorer,
    ThresholdValidator,
    UniformWeighter,
    YesToAllScorer,
)
from scripts.ensemble.learning import (
    IsolatedPredictor,
    ModelTrainer,
    PredictiveEnsembler,
    TrainedPredictor,
)
from scripts.ensemble.optimization import (
    get_custom_ensembler,
    get_trained_predictor,
    optimize_sampler_fn,
)
from scripts.ensemble.utils import (
    exclude_from,
    extract_submissions,
    keep_annotated_sentences,
    keep_best_per_participant,
    keep_non_annotated_sentences,
    keep_top_k_submissions,
)
from scripts.extract import (
    normalize_scores,
    performance_per_agreement,
    plot_performance,
    sort_sentences,
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


def get_f1_ensembler(choir: EnsembleChoir, binary: bool):
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


def get_participant_ensembler(choir: EnsembleChoir, binary: bool, name: str):
    orchestrator = EnsembleOrchestrator(binary=binary)

    weighter = UniformWeighter.build()
    scorer = FocusedScorer(name, discrete=True)
    validator = NonZeroValidator()

    ensembler = ManualVotingEnsembler(choir, orchestrator, weighter, scorer, validator)
    return ensembler


def get_union_ensembler(choir: EnsembleChoir, binary: bool):
    orchestrator = EnsembleOrchestrator(binary=binary)

    weighter = UniformWeighter.build()
    scorer = YesToAllScorer()
    validator = NonZeroValidator()

    ensembler = ManualVotingEnsembler(choir, orchestrator, weighter, scorer, validator)
    return ensembler


def get_sklearn_ensembler(
    choir: EnsembleChoir,
    model_type,
    mode: Literal["category", "all", "each"],
    ignore=(),
    weighting_table=None,
    **kargs,
):
    orchestrator = EnsembleOrchestrator(binary=True)

    reference = orchestrator(choir)
    predictor = get_trained_predictor(
        reference,
        model_type,
        mode=mode,
        ignore=ignore,
        weighting_table=weighting_table,
        **kargs,
    )

    ensembler = PredictiveEnsembler(choir, orchestrator, predictor)
    return ensembler


def get_isolated_ensembler(
    choir: EnsembleChoir,
    taskA_choir: EnsembleChoir,
    taskB_choir: EnsembleChoir,
    model_type,
    mode: Literal["category", "all", "each"],
    ignore=(),
    weighting_table=None,
    **kargs,
):
    orchestrator = EnsembleOrchestrator(binary=True)

    taskA_predictor = get_trained_predictor(
        orchestrator(taskA_choir),
        model_type,
        mode=mode,
        ignore=ignore,
        weighting_table=weighting_table,
        **kargs,
    )
    taskB_predictor = get_trained_predictor(
        orchestrator(taskB_choir),
        model_type,
        mode=mode,
        ignore=ignore,
        weighting_table=weighting_table,
        **kargs,
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


def task_run(ensembler: Ensembler, target: EnsembleChoir, scenario="1-main"):
    ensembled = ensembler(target)
    print(
        "==== SCORE ====\n",
        EnsembleChoir.evaluate(
            ensembled,
            target.gold,
            clamp=True,
            skipA=scenario == "3-taskB",
            skipB=scenario == "2-taskA",
        ),
    )


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
        (sname, s)
        for sname, s in choir.submissions.items()
        if sname.split("/")[0] == name
    ]
    assert len(submissions) == 1
    selected_name, submit = submissions[0]
    print(selected_name)
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
    CollectionV1Handler.dump(
        selection.gold, output / "input_scenario1.txt", skip_empty_sentences=False
    )

    for name, submit in selection.submissions.items():
        output: Path = outputdir / "submissions" / name / "scenario1-main"
        output.mkdir(parents=True)
        CollectionV1Handler.dump(
            submit, output / "output_scenario1.txt", skip_empty_sentences=False
        )


def task_performance_per_agreement(
    ensembler: Ensembler,
    score=False,
    linestyle="-",
    marker=None,
    alpha=1.0,
    **targets: EnsembleChoir,
):
    sequences = {}

    for label, target in targets.items():
        target = keep_annotated_sentences(target)
        ensembled = ensembler(target, collection_only=False)

        for direction in ["top higher", "top lower"]:
            sequence = []
            for item in performance_per_agreement(
                ensembled, normalized=True, reverse=(direction == "top higher")
            ):
                print(item)
                sequence.append(item)

            sequences[f"{label} ({direction})"] = sequence

    if score:
        plot_performance(
            sequences,
            order=["score", "top", "f1"],
            linestyle=linestyle,
            marker=marker,
            alpha=alpha,
        )
    else:
        plot_performance(sequences, linestyle=linestyle, marker=marker, alpha=alpha)


def task_do_ensemble(
    ensembler: Ensembler, target: EnsembleChoir, top=None, opath: Path = None
):
    ensembled = ensembler(target, collection_only=False)
    sentences = []
    scores = []
    for sid, _, score in normalize_scores(
        sort_sentences(ensembled, reverse=True), target
    ):
        sentence = ensembled.collection.sentences[sid]
        if sentence.annotated:
            sentences.append(sentence)
            scores.append(score)

    sentences = sentences if top is None else sentences[:top]
    scores = scores if top is None else scores[:top]

    output_collection = Collection(sentences)

    opath = Path("./data/ensemble/ensemble.txt") if opath is None else opath
    CollectionV2Handler.dump(output_collection, opath)
    scr_path: Path = opath.parent / (opath.stem + ".scr")
    scr_path.write_text("\n".join(str(s) for s in scores))


if __name__ == "__main__":
    path2ehealth19_submissions = Path("./data/ehealth2019/submissions/all")
    path2ehealth19_gold = Path("./data/ehealth2019/testing")
    path2scenario1 = path2ehealth19_gold / "scenario1-main"
    path2scenario2 = path2ehealth19_gold / "scenario2-taskA"
    path2scenario3 = path2ehealth19_gold / "scenario3-taskB"
    path2ehealth20_submissions = Path("./data/ehealth2020/submissions")
    path2ehealth20_gold = Path("./data/ehealth2020/testing")
    path2scenario4 = path2ehealth20_gold / "scenario1-main"
    path2ehealth20_ann = Path("./data/ehealth2020/ann")

    # task_extract(
    #     Path("./data/ehealth2020"),
    #     path2ehealth19_submissions,
    #     path2ehealth19_gold,
    #     path2ehealth20_ann,
    # )

    scenario = ["1-main", "2-taskA", "3-taskB"][1]
    ref_submissions = path2ehealth20_submissions
    ref_test = path2ehealth20_gold
    val_submissions = path2ehealth19_submissions
    val_test = path2ehealth19_gold

    # ref_submissions = path2ehealth19_submissions
    # ref_test = path2ehealth19_gold
    # val_submissions = path2ehealth20_submissions
    # val_test = path2ehealth20_gold

    print(f"======== Loading ... (reference) ===============")
    choir = EnsembleChoir().load(CollectionV1Handler, ref_submissions, ref_test)
    print("======== Done! =================================")

    # ehealth2020_gold = CollectionV1Handler.load(
    #     Collection(), path2ehealth20_gold / "scenario1-main" / "input_scenario1.txt"
    # )
    # choir.gold = choir.gold.complete_from(ehealth2020_gold)
    # validation = choir

    print(f"======== Loading ... (validation) ==============")
    validation = EnsembleChoir().load(
        CollectionV1Handler, val_submissions, val_test, scenario=scenario
    )
    print("======== Done! =================================")

    # # BEST
    # choir = keep_best_per_participant(choir)
    # validation = keep_best_per_participant(validation)

    # # TOP
    # choir = keep_top_k_submissions(choir, None)
    # validation = keep_top_k_submissions(validation, None)

    # ensembler = get_majority_ensembler(choir, binary=True)
    # ensembler = get_f1_ensembler(choir, binary=True)
    # ensembler = get_participant_ensembler(choir, binary=True, name="talp/576640")
    # ensembler = get_union_ensembler(choir, binary=False)
    # ensembler = get_gold_ensembler(choir, binary=True)
    # ensembler = get_gold_ensembler(validation, binary=True)
    # ensembler = get_sklearn_ensembler(choir, model_type=SVC, mode="all")
    # ensembler = get_sklearn_ensembler(
    #     choir,
    #     model_type=RandomForestClassifier,
    #     mode="category",
    #     weighting_table=F1Weighter.build(choir).table,
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
    # ensembler = get_custom_ensembler(
    #     choir,
    #     choir.gold_annotated,
    #     {
    #         "load-best": True,
    #         "learning": True,
    #         "model-type": "randf",
    #         "training-mode": "category",
    #         "weight-learning-votes": True,
    #     },
    #     manual_voting=True,
    #     learning=True,
    # )

    # task_run(ensembler, choir, scenario=scenario)
    # task_run(ensembler, validation, scenario=scenario)
    # task_validate_submission(validation, "talp", validation.gold)
    # task_performance_per_agreement(
    #     ensembler, score=False, reference=choir, validation=validation, alpha=0.7,
    # )
    # best, best_fn = optimize_sampler_fn(
    #     choir,
    #     choir.gold_annotated,
    #     generations=500,
    #     pop_size=10,
    #     manual_voting=True,
    #     learning=True,
    # )
    # ensembler = best.model
    # task_cross_validate(
    #     choir,
    #     taskA_choir,
    #     taskB_choir,
    #     model_type=RandomForestClassifier,
    #     mode="category",
    #     limit=None,
    # )

    # print("Reference score:", ensembler.choir.eval(ensembler()))
    # print(
    #     "Validation score:",
    #     validation.eval(
    #         ensembler(validation),
    #         skipA=(scenario == "3-taskB"),
    #         skipB=(scenario == "2-taskA"),
    #     ),
    # )

    # print(len(validation.gold), len(validation.gold_annotated))
    # output = keep_non_annotated_sentences(validation)
    # print(len(output.gold))
    # print(len(choir.gold), len(choir.gold_annotated))
    # output = exclude_from(output, choir.gold_annotated)
    # print(len(output.gold))
    # task_do_ensemble(ensembler, output, top=3000)
    # Hay oraciones que estan tanto en el test viejo como en el nuevo!!!
