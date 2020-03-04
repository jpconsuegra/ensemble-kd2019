from pathlib import Path

from scripts.ensemble import EnsembleChoir, EnsembleOrchestrator, Ensembler
from scripts.ensemble.ensemblers import (
    AggregateTopScorer,
    AverageScorer,
    AverageTopScorer,
    ConstantThresholdValidator,
    ExpertScorer,
    F1Weighter,
    GoldOracleScorer,
    ManualVotingEnsembler,
    MaxScorer,
    NonZeroValidator,
    ThresholdValidator,
    UniformWeighter,
)
from scripts.ensemble.utils import keep_top_k_submissions
from scripts.ensemble.optimization import optimize_parametric_fn, optimize_sampler_fn

# from statistics import mean, quantiles, stdev, variance


# def validate_model(
#     submits: Path,
#     gold: Path,
#     *,
#     best=True,
#     limit=None,
#     model_type=RandomForestClassifier,
#     model_handler_init=AllInOneModel,
# ):

#     ensemble = MultiSourceEnsemble(
#         model_type=model_type, model_handler_init=model_handler_init
#     )
#     ensemble.load(submits, gold, best=best)

#     selected_sids = [
#         x for x in ensemble.gold_annotated_sid() if x < ensemble.load_stack[0]
#     ]
#     if limit is not None:
#         selected_sids = selected_sids[:limit]

#     with open("log.txt", mode="w") as fd:
#         scores = []

#         for sid in tqdm(selected_sids):
#             ensemble.build(ignore=(sid,))
#             ensemble.make()
#             ensembled_collection = Collection([ensemble.collection.sentences[sid]])
#             gold_collection = Collection([ensemble.gold.sentences[sid]])
#             score = ensemble.evaluate(ensembled_collection, gold_collection)
#             scores.append(score)
#             fd.write(f"{sid} -> {score}\n")
#             fd.flush()

#         return scores


# def task_validate(ps, pg, best, model_type):
#     scores = validate_model(ps, pg, best=best, model_type=model_type)
#     print("\n".join(str(x) for x in scores))
#     print("=================================")
#     print("|= mean ======", mean(scores))
#     print("|= stdev =====", stdev(scores))
#     print("|= variance ==", variance(scores))
#     print("|= quantiles =", quantiles(scores))
#     print("=================================")


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
    scorer = ExpertScorer(weighter, choir)
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


def task_run(ensembler: Ensembler):
    ensembled = ensembler()
    print("==== SCORE ====\n", ensembled.choir.eval(ensembled))


if __name__ == "__main__":

    ps = Path("./data/submissions/all")
    pg = Path("./data/testing")
    choir = EnsembleChoir().load(ps, pg, best=False)

    # ensembler = get_f1_ensembler(choir, binary=True)

    # e = SklearnEnsemble()
    # e = SklearnEnsemble(model_handler_init=PerLabelModel)
    # e = IsolatedDualEnsemble()
    # e = IsolatedDualEnsemble(model_handler_init=PerLabelModel)
    # e = MultiScenarioSKEmsemble()
    # e = MultiScenarioSKEmsemble(model_handler_init=PerLabelModel)
    # e = MultiSourceEnsemble()
    # e = MultiSourceEnsemble(model_type=SVC, model_handler_init=PerLabelModel)

    # task_run(ensembler)
    # optimize_parametric_fn(choir, generations=30)
    # optimize_sampler_fn(choir, generations=100, show_model=True)
    # task_validate(ps, pg, best=False, model_type=AllInOneModel)
