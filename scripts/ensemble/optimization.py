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
    ManualVotingEnsembler,
    MaxScorer,
    NonZeroValidator,
    ThresholdValidator,
    UniformWeighter,
)
from scripts.ensemble.utils import keep_named_submissions, keep_top_k_submissions
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

    soft_cache = {"F1Weighter": F1Weighter.build(choir)}

    def generator(sampler: Sampler):

        sampler = LogSampler(sampler)

        # ---- ORCHESTRATOR ---------------------------------------------
        binary = sampler.boolean("binary")
        orchestrator = EnsembleOrchestrator(binary=binary)

        # ---- TRAINING CHOIR -------------------------------------------
        if sampler.boolean("top-best"):
            n_submits = sampler.discrete(1, len(choir.submissions), "top-submits")
            training_choir = keep_top_k_submissions(choir, n_submits)
        else:
            n_submits = sampler.discrete(1, len(choir.submissions), "n-submits")
            submissions = sampler.multichoice(
                choir.submissions.keys(), n_submits, "submissions"
            )
            training_choir = keep_named_submissions(choir, submissions)

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
            scorer = ExpertScorer(weighter, training_choir)
        elif scorer == "max":
            scorer = MaxScorer()
        elif scorer == "avg-top":
            k = sampler.discrete(1, n_submits, "k-avg-top")
            strict = sampler.boolean("strict")
            scorer = AverageTopScorer(k, strict)
        elif scorer == "sum-top":
            k = sampler.discrete(1, n_submits, "k-sum-top")
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
            thresholds = sampler.multisample(
                ENTITIES + RELATIONS,
                sampler.continuous,
                handle="thresholds",
                min=0,
                max=1,
            )
            validator = ThresholdValidator(thresholds)
        elif validator == "constant":
            threshold = sampler.continuous(0, 1, "threshold")
            validator = ConstantThresholdValidator(threshold)
        else:
            raise Exception()

        # ==== ENSEMBLER ================================================
        ensembler = ManualVotingEnsembler(
            training_choir, orchestrator, weighter, scorer, validator
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
        allow_duplicates=False,
        pop_size=pop_size,
    )
    loggers = [ProgressLogger(), ConsoleLogger()]
    best, best_fn = search.run(generations=generations, logger=loggers)
    return best, best_fn
