from autogoal import optimize
from autogoal.grammar import Boolean, Categorical, Continuous, Discrete
from autogoal.sampling import Sampler
from autogoal.search import ConsoleLogger, PESearch, ProgressLogger, Logger
from autogoal.utils import nice_repr

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
from scripts.utils import ENTITIES, RELATIONS


@nice_repr
class ThresholdMap:
    def __init__(
        self,
        concept_threshold: Continuous(0, 1),
        action_threshold: Continuous(0, 1),
        predicate_threshold: Continuous(0, 1),
        reference_threshold: Continuous(0, 1),
        is_a_threshold: Continuous(0, 1),
        same_as_threshold: Continuous(0, 1),
        part_of_threshold: Continuous(0, 1),
        has_property_threshold: Continuous(0, 1),
        causes_threshold: Continuous(0, 1),
        entails_threshold: Continuous(0, 1),
        in_context_threshold: Continuous(0, 1),
        in_place_threshold: Continuous(0, 1),
        in_time_threshold: Continuous(0, 1),
        subject_threshold: Continuous(0, 1),
        target_threshold: Continuous(0, 1),
        domain_threshold: Continuous(0, 1),
        arg_threshold: Continuous(0, 1),
    ):
        self.concept_threshold = concept_threshold
        self.action_threshold = action_threshold
        self.predicate_threshold = predicate_threshold
        self.reference_threshold = reference_threshold
        self.is_a_threshold = is_a_threshold
        self.same_as_threshold = same_as_threshold
        self.part_of_threshold = part_of_threshold
        self.has_property_threshold = has_property_threshold
        self.causes_threshold = causes_threshold
        self.entails_threshold = entails_threshold
        self.in_context_threshold = in_context_threshold
        self.in_place_threshold = in_place_threshold
        self.in_time_threshold = in_time_threshold
        self.subject_threshold = subject_threshold
        self.target_threshold = target_threshold
        self.domain_threshold = domain_threshold
        self.arg_threshold = arg_threshold

        self.thresholds = {
            "Concept": concept_threshold,
            "Action": action_threshold,
            "Predicate": predicate_threshold,
            "Reference": reference_threshold,
            "is-a": is_a_threshold,
            "same-as": same_as_threshold,
            "part-of": part_of_threshold,
            "has-property": has_property_threshold,
            "causes": causes_threshold,
            "entails": entails_threshold,
            "in-context": in_context_threshold,
            "in-place": in_place_threshold,
            "in-time": in_time_threshold,
            "subject": subject_threshold,
            "target": target_threshold,
            "domain": domain_threshold,
            "arg": arg_threshold,
        }

    def __iter__(self):
        return iter(self.thresholds.items())

    def __getitem__(self, label):
        return self.thresholds[label]


def build_parametric_fn(choir: EnsembleChoir):
    orchestrators = {
        "ordinary": EnsembleOrchestrator(binary=False),
        "binary": EnsembleOrchestrator(binary=True),
    }
    weighters = {"uniform": UniformWeighter.build(), "f1": F1Weighter.build(choir)}
    scorers = {
        "avg": lambda **kargs: AverageScorer(),
        "expert": lambda **kargs: ExpertScorer(weighter=kargs["weighter"], choir=choir),
        "max": lambda **kargs: MaxScorer(),
        "avg-top": lambda **kargs: AverageTopScorer(
            k=kargs["k"], strict=kargs["strict"]
        ),
        "sum-top": lambda **kargs: AggregateTopScorer(k=kargs["k"]),
    }
    validators = {
        "non-zero": lambda **kargs: NonZeroValidator(),
        "threshold": lambda **kargs: ThresholdValidator(thresholds=kargs["thresholds"]),
    }

    def build_and_score_ensemble(
        orchestrator: Categorical(*orchestrators.keys()),
        weighter: Categorical(*weighters.keys()),
        scorer: Categorical(*scorers.keys()),
        k: Discrete(1, len(choir.submissions)),
        strict: Boolean(),
        validator: Categorical(*validators.keys()),
        thresholds: ThresholdMap,
    ):
        orchestrator = orchestrators[orchestrator]
        weighter = weighters[weighter]
        scorer = scorers[scorer](weighter=weighter, k=k, strict=strict)
        validator = validators[validator](thresholds=thresholds)

        ensembler = ManualVotingEnsembler(
            choir, orchestrator, weighter, scorer, validator
        )
        ensembled = ensembler()
        return choir.eval(ensembled)

    return build_and_score_ensemble


class Cache:
    def __init__(self):
        self.memory = {}

    def get(self, tag, default, **kargs):
        key = tuple(kargs.items())
        try:
            return self.memory[tag, key]
        except KeyError:
            value = self.memory[tag, key] = default()
            return value


def build_sampler_fn(choir: EnsembleChoir):

    soft_cache = {"F1Weighter": F1Weighter.build(choir)}

    def from_sampler_fn(sampler: Sampler):

        # ---- ORCHESTRATOR ---------------------------------------------
        binary = sampler.boolean("binary")
        orchestrator = EnsembleOrchestrator(binary=binary)

        # ---- WEIGHTER -------------------------------------------------
        weighter = sampler.categorical(["uniform", "f1"], "weighter")
        if weighter == "uniform":
            weighter = UniformWeighter.build()
        elif weighter == "f1":
            weighter = soft_cache["F1Weighter"]  # F1Weighter.build(choir)
        else:
            raise Exception()

        # ---- SCORER ---------------------------------------------------
        scorer = sampler.categorical(
            ["avg", "expert", "max", "avg-top", "sum-top"], "scorer"
        )
        if scorer == "avg":
            scorer = AverageScorer()
        elif scorer == "expert":
            scorer = ExpertScorer(weighter, choir)
        elif scorer == "max":
            scorer = MaxScorer()
        elif scorer == "avg-top":
            k = sampler.discrete(1, len(choir.submissions), "k-avg-top")
            strict = sampler.boolean("strict")
            scorer = AverageTopScorer(k, strict)
        elif scorer == "sum-top":
            k = sampler.discrete(1, len(choir.submissions), "k-sum-top")
            scorer = AggregateTopScorer(k)
        else:
            raise Exception()

        # ---- VALIDATOR ------------------------------------------------
        validator = sampler.categorical(
            ["non-zero", "threshold", "constant"], "validator"
        )
        if validator == "non-zero":
            validator = NonZeroValidator()
        elif validator == "threshold":
            thresholds = {
                label: sampler.continuous(0, 1, f"threshold-{label}")
                for label in ENTITIES + RELATIONS
            }
            validator = ThresholdValidator(thresholds)
        elif validator == "constant":
            threshold = sampler.continuous(0, 1, "threshold")
            validator = ConstantThresholdValidator(threshold)
        else:
            raise Exception()

        # ==== ENSEMBLER ================================================
        ensembler = ManualVotingEnsembler(
            choir, orchestrator, weighter, scorer, validator
        )

        ensembled = ensembler()
        return choir.eval(ensembled)

    return from_sampler_fn


class MyDict(dict):
    pass


def build_generator_and_fn(choir: EnsembleChoir):
    def generator(sampler: Sampler):
        binary = sampler.boolean("binary")
        weighter = sampler.categorical(["uniform", "f1"], "weighter")
        scorer = sampler.categorical(
            ["avg", "expert", "max", "avg-top", "sum-top"], "scorer"
        )
        validator = sampler.categorical(
            ["non-zero", "threshold", "constant"], "validator"
        )

        k_avg_top = (
            sampler.discrete(1, len(choir.submissions), "k-avg-top")
            if scorer == "avg-top"
            else None
        )
        strict = sampler.boolean("strict") if scorer == "avg-top" else None
        k_sum_top = (
            sampler.discrete(1, len(choir.submissions), "k-sum-top")
            if scorer == "sum-top"
            else None
        )

        thresholds = (
            {
                label: sampler.continuous(0, 1, f"threshold-{label}")
                for label in ENTITIES + RELATIONS
            }
            if validator == "threshold"
            else None
        )

        threshold = (
            sampler.continuous(0, 1, "threshold") if validator == "constant" else None
        )

        return MyDict(
            {
                "binary": binary,
                "weighter": weighter,
                "scorer": scorer,
                "validator": validator,
                "k-avg-top": k_avg_top,
                "strict": strict,
                "k-sum-top": k_sum_top,
                "thresholds": thresholds,
                "threshold": threshold,
            }
        )

    soft_cache = {"F1Weighter": F1Weighter.build(choir)}

    def fn(params):

        # ---- ORCHESTRATOR ---------------------------------------------
        binary = params["binary"]
        orchestrator = EnsembleOrchestrator(binary=binary)

        # ---- WEIGHTER -------------------------------------------------
        weighter = params["weighter"]
        if weighter == "uniform":
            weighter = UniformWeighter.build()
        elif weighter == "f1":
            weighter = soft_cache["F1Weighter"]  # F1Weighter.build(choir)
        else:
            raise Exception()

        # ---- SCORER ---------------------------------------------------
        scorer = params["scorer"]
        if scorer == "avg":
            scorer = AverageScorer()
        elif scorer == "expert":
            scorer = ExpertScorer(weighter, choir)
        elif scorer == "max":
            scorer = MaxScorer()
        elif scorer == "avg-top":
            k = params["k-avg-top"]
            strict = params["strict"]
            scorer = AverageTopScorer(k, strict)
        elif scorer == "sum-top":
            k = params["k-sum-top"]
            scorer = AggregateTopScorer(k)
        else:
            raise Exception()

        # ---- VALIDATOR ------------------------------------------------
        validator = params["validator"]
        if validator == "non-zero":
            validator = NonZeroValidator()
        elif validator == "threshold":
            thresholds = params["thresholds"]
            validator = ThresholdValidator(thresholds)
        elif validator == "constant":
            threshold = params["threshold"]
            validator = ConstantThresholdValidator(threshold)
        else:
            raise Exception()

        # ==== ENSEMBLER ================================================
        ensembler = ManualVotingEnsembler(
            choir, orchestrator, weighter, scorer, validator
        )

        ensembled = ensembler()
        return choir.eval(ensembled)

    return generator, fn


def optimize_parametric_fn(choir: EnsembleChoir, generations):
    fn = build_parametric_fn(choir)
    loggers = [ProgressLogger(), ConsoleLogger()]
    print(optimize(fn, allow_duplicates=False, logger=loggers, generations=generations))


def optimize_sampler_fn(choir: EnsembleChoir, generations, pop_size, show_model=True):
    if show_model:
        generator, fn = build_generator_and_fn(choir)
        search = PESearch(
            generator_fn=generator,
            fitness_fn=fn,
            evaluation_timeout=0,
            memory_limit=0,
            allow_duplicates=False,
            pop_size=pop_size,
        )
    else:
        fn = build_sampler_fn(choir)
        search = PESearch(
            fitness_fn=fn,
            evaluation_timeout=0,
            memory_limit=0,
            allow_duplicates=True,
            pop_size=pop_size,
        )
    loggers = [ProgressLogger(), ConsoleLogger()]
    best, best_fn = search.run(generations=generations, logger=loggers)
    print((format_history(best._history) if not show_model else best), best_fn)


def format_history(history) -> str:
    return str({record["args"]: record["result"] for record in history})
