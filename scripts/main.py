from pathlib import Path

from scripts.ensemble import EnsembleChoir, EnsembleOrchestrator, Ensembler
from scripts.ensemble.ensemblers import (
    AverageScorer,
    ConstantThresholdValidator,
    ExpertScorer,
    F1Weighter,
    GoldOracleScorer,
    ManualVotingEnsembler,
    NonZeroValidator,
    ThresholdValidator,
    UniformWeighter,
)
from scripts.ensemble.utils import keep_top_k_submissions

# from statistics import mean, quantiles, stdev, variance

# from autogoal import optimize
# from autogoal.grammar import Continuous
# from autogoal.search import ConsoleLogger, ProgressLogger
# from autogoal.utils import nice_repr

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


# @nice_repr
# class ThresholdMap:
#     def __init__(
#         self,
#         concept_threshold: Continuous(0, 1),
#         action_threshold: Continuous(0, 1),
#         predicate_threshold: Continuous(0, 1),
#         reference_threshold: Continuous(0, 1),
#         is_a_threshold: Continuous(0, 1),
#         same_as_threshold: Continuous(0, 1),
#         part_of_threshold: Continuous(0, 1),
#         has_property_threshold: Continuous(0, 1),
#         causes_threshold: Continuous(0, 1),
#         entails_threshold: Continuous(0, 1),
#         in_context_threshold: Continuous(0, 1),
#         in_place_threshold: Continuous(0, 1),
#         in_time_threshold: Continuous(0, 1),
#         subject_threshold: Continuous(0, 1),
#         target_threshold: Continuous(0, 1),
#         domain_threshold: Continuous(0, 1),
#         arg_threshold: Continuous(0, 1),
#     ):
#         self.concept_threshold = concept_threshold
#         self.action_threshold = action_threshold
#         self.predicate_threshold = predicate_threshold
#         self.reference_threshold = reference_threshold
#         self.is_a_threshold = is_a_threshold
#         self.same_as_threshold = same_as_threshold
#         self.part_of_threshold = part_of_threshold
#         self.has_property_threshold = has_property_threshold
#         self.causes_threshold = causes_threshold
#         self.entails_threshold = entails_threshold
#         self.in_context_threshold = in_context_threshold
#         self.in_place_threshold = in_place_threshold
#         self.in_time_threshold = in_time_threshold
#         self.subject_threshold = subject_threshold
#         self.target_threshold = target_threshold
#         self.domain_threshold = domain_threshold
#         self.arg_threshold = arg_threshold

#         self.thresholds = {
#             "Concept": concept_threshold,
#             "Action": action_threshold,
#             "Predicate": predicate_threshold,
#             "Reference": reference_threshold,
#             "is-a": is_a_threshold,
#             "same-as": same_as_threshold,
#             "part-of": part_of_threshold,
#             "has-property": has_property_threshold,
#             "causes": causes_threshold,
#             "entails": entails_threshold,
#             "in-context": in_context_threshold,
#             "in-place": in_place_threshold,
#             "in-time": in_time_threshold,
#             "subject": subject_threshold,
#             "target": target_threshold,
#             "domain": domain_threshold,
#             "arg": arg_threshold,
#         }

#     def __iter__(self):
#         return iter(self.thresholds.items())


# def build_fn(ensemble: Ensemble):
#     def fn(thresholds: ThresholdMap):
#         e = SetParameterAsThreshold(ensemble, thresholds)
#         e.build()
#         e.make()
#         return e.eval()

#     return fn


# def task_optimize(e: Ensemble, best, iterations):
#     e.load(ps, pg, best=best)
#     loggers = [ProgressLogger(), ConsoleLogger()]
#     print(optimize(build_fn(e), logger=loggers, iterations=iterations))


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

    ensembler = get_f1_ensembler(choir, binary=True)

    # e = SklearnEnsemble()
    # e = SklearnEnsemble(model_handler_init=PerLabelModel)
    # e = IsolatedDualEnsemble()
    # e = IsolatedDualEnsemble(model_handler_init=PerLabelModel)
    # e = MultiScenarioSKEmsemble()
    # e = MultiScenarioSKEmsemble(model_handler_init=PerLabelModel)
    # e = MultiSourceEnsemble()
    # e = MultiSourceEnsemble(model_type=SVC, model_handler_init=PerLabelModel)

    task_run(ensembler)
    # task_optimize(e, best=False, iterations=10)
    # task_validate(ps, pg, best=False, model_type=AllInOneModel)
