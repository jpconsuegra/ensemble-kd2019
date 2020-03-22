from functools import total_ordering

from autogoal import optimize
from autogoal.grammar import Boolean, Categorical, Continuous, Discrete
from autogoal.sampling import Sampler
from autogoal.search import ConsoleLogger, Logger, PESearch, ProgressLogger
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
    MajorityValidator,
    ManualVotingEnsembler,
    MaxScorer,
    NonZeroValidator,
    SumScorer,
    ThresholdValidator,
    UniformWeighter,
)
from scripts.ensemble.utils import (
    keep_best_per_participant,
    keep_named_submissions,
    keep_top_k_submissions,
)
from scripts.utils import ENTITIES, RELATIONS


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


class LogSampler:
    def __init__(self, sampler: Sampler):
        self._sampler = sampler
        self._log = []

    def _log_sample(self, handle, value, log=True):
        result = value
        if log:
            if handle is None:
                handle, value = value, True
            self._log.append((handle, value))
        return result

    def boolean(self, handle=None, log=True) -> bool:
        value = self._sampler.boolean(handle)
        return self._log_sample(handle, value, log=log)

    def categorical(self, options, handle=None, log=True):
        value = self._sampler.categorical(options, handle)
        return self._log_sample(handle, value, log=log)

    def choice(self, options, handle=None, log=True):
        value = self._sampler.choice(options, handle)
        return self._log_sample(handle, value, log=log)

    def continuous(self, min=0, max=1, handle=None, log=True) -> float:
        value = self._sampler.continuous(min, max, handle)
        return self._log_sample(handle, value, log=log)

    def discrete(self, min=0, max=10, handle=None, log=True) -> int:
        value = self._sampler.discrete(min, max, handle)
        return self._log_sample(handle, value, log=log)

    def multichoice(self, options, k, handle=None, log=True):
        values = []
        candidates = list(options)
        for _ in range(k):
            value = self._sampler.choice(candidates)
            candidates.remove(value)
            values.append(value)
        return self._log_sample(handle, values, log=log)

    def multisample(self, labels, func, handle=None, log=True, **kargs):
        values = {
            label: func(handle=f"{handle}-{label}", log=False, **kargs)
            for label in labels
        }
        return self._log_sample(handle, values, log=log)

    def __iter__(self):
        return iter(self._log)

    def __str__(self):
        items = ",\n    ".join(f"{k}: {v}" for k, v in self)
        return "{\n    " + items + "\n}"

    def __repr__(self):
        return str(self)


class SampleModel:
    def __init__(self, sampler, model):
        self.sampler = sampler
        self.model = model

    def __str__(self):
        return str(self.sampler)

    def __repr__(self):
        return repr(self.sampler)


def build_generator_and_fn(choir: EnsembleChoir):

    print("======== Caching ... F1Weighter =============")
    cached_f1_weighter = F1Weighter.build(choir)
    print("======== Caching ... Best choir =============")
    cached_best_choir = keep_best_per_participant(choir)

    def generator(sampler: Sampler):

        sampler = LogSampler(sampler)
        train_choir = choir

        # ---- ORCHESTRATOR ---------------------------------------------
        binary = sampler.boolean("binary")
        orchestrator = EnsembleOrchestrator(binary=binary)

        # ---- TRAINING CHOIR -------------------------------------------
        if sampler.boolean("load-best"):
            train_choir = cached_best_choir

        if sampler.boolean("top-best"):
            n_submits = sampler.discrete(1, len(train_choir.submissions), "top-submits")
            train_choir = keep_top_k_submissions(train_choir, n_submits)
        else:
            n_submits = sampler.discrete(1, len(train_choir.submissions), "n-submits")
            submissions = sampler.multichoice(
                train_choir.submissions.keys(), n_submits, "submissions"
            )
            submissions = sorted(submissions)  # avoid duplicated models
            train_choir = keep_named_submissions(train_choir, submissions)

        # ---- WEIGHTER -------------------------------------------------
        _weighter = sampler.categorical(["uniform", "f1"], "weighter")
        if _weighter == "uniform":
            weighter = UniformWeighter.build()
        elif _weighter == "f1":
            weighter = cached_f1_weighter  # F1Weighter.build(choir)
        else:
            raise Exception()

        # ---- SCORER ---------------------------------------------------
        _scorer = sampler.categorical(
            ["avg", "sum", "expert", "max", "avg-top", "sum-top"], "scorer"
        )
        if _scorer == "avg":
            scorer = AverageScorer()
        elif _scorer == "sum":
            scorer = SumScorer()
        elif _scorer == "expert":
            discrete = sampler.boolean("discrete-expert")
            scorer = ExpertScorer(weighter, train_choir, discrete)
        elif _scorer == "max":
            scorer = MaxScorer()
        elif _scorer == "avg-top":
            k = sampler.discrete(1, n_submits, "k-avg-top")
            strict = sampler.boolean("strict")
            scorer = AverageTopScorer(k, strict)
        elif _scorer == "sum-top":
            k = sampler.discrete(1, n_submits, "k-sum-top")
            scorer = AggregateTopScorer(k)
        else:
            raise Exception()

        # ---- VALIDATOR ------------------------------------------------
        _validator = sampler.categorical(
            ["non-zero", "threshold", "constant", "majority"], "validator"
        )
        if _validator == "non-zero":
            validator = NonZeroValidator()
        elif _validator == "threshold":
            if sampler.boolean("use-disc-thresholds"):  # _weighter == "uniform":
                thresholds = sampler.multisample(
                    ENTITIES + RELATIONS,
                    sampler.discrete,
                    handle="disc-thresholds",
                    min=0,
                    max=n_submits,
                )
            else:
                thresholds = sampler.multisample(
                    ENTITIES + RELATIONS,
                    sampler.continuous,
                    handle="cont-thresholds",
                    min=0,
                    max=1,
                )
            validator = ThresholdValidator(thresholds)
        elif _validator == "constant":
            threshold = (
                sampler.discrete(0, n_submits, "disc-threshold")
                if sampler.boolean("use-disc-threshold")  # if _weighter == "uniform"
                else sampler.continuous(0, 1, "cont-threshold")
            )
            validator = ConstantThresholdValidator(threshold)
        elif _validator == "majority":
            validator = MajorityValidator(n_submits)
        else:
            raise Exception()

        # ==== ENSEMBLER ================================================
        ensembler = ManualVotingEnsembler(
            train_choir, orchestrator, weighter, scorer, validator
        )
        return SampleModel(sampler, ensembler)

    def fn(generated: SampleModel):
        ensembler = generated.model
        ensembled = ensembler()
        return choir.eval(ensembled)

    return generator, fn


def optimize_sampler_fn(choir: EnsembleChoir, generations, pop_size):
    generator, fn = build_generator_and_fn(choir)
    search = PESearch(
        generator_fn=generator,
        fitness_fn=fn,
        evaluation_timeout=0,
        memory_limit=0,
        search_timeout=0,
        allow_duplicates=False,
        pop_size=pop_size,
    )
    loggers = [ProgressLogger(), ConsoleLogger()]
    best, best_fn = search.run(generations=generations, logger=loggers)
    return best, best_fn
