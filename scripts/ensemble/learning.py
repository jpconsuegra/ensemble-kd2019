from typing import Generator, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tqdm import tqdm

from scripts.ensemble import EnsembleChoir, EnsembledCollection, Ensembler
from scripts.ensemble.features import (
    ModelHandler,
    model_handler_assistant,
)
from scripts.utils import ENTITIES, RELATIONS, Keyphrase, Relation


class ModelTrainer:
    def __init__(self, model_handler_init):
        self._model_handler_init = model_handler_init

    def __call__(
        self, choir: EnsembleChoir, annotation_votes, ignore=()
    ) -> ModelHandler:
        return self.build_and_train_model_handler(
            self._model_handler_init, choir, annotation_votes, ignore,
        )

    @classmethod
    def build_and_train_model_handler(
        cls, model_handler_init, choir: EnsembleChoir, annotation_votes, ignore=(),
    ) -> ModelHandler:
        selected_sids = {x for x in choir.gold_annotated_sid() if x not in ignore}

        model_handler = model_handler_init()

        for tag, (model, sids, anns, labels, X) in model_handler(
            annotation_votes, selected_sids=selected_sids
        ).items():
            y = cls._build_targets(choir, anns, labels, sids)
            model.fit(X, y)
            print(f"Training score ({tag}):", model.score(X, y))

        return model_handler

    @classmethod
    def _build_targets(cls, choir: EnsembleChoir, annotations, labels, sids):
        targets = []
        for ann, label, sid in zip(annotations, labels, sids):
            gold_sentence = choir.gold.sentences[sid]
            gold_annotation = gold_sentence.find_first_match(ann, label=label)
            targets.append(int(gold_annotation is not None))
        return np.asarray(targets)


class Predictor:
    def __call__(
        self, annotation_votes, task
    ) -> Generator[Tuple[Union[Keyphrase, Relation], Optional[str]], None, None]:
        pass


class TrainedPredictor(Predictor):
    def __init__(
        self,
        reference_collection: EnsembledCollection,
        threshold: float,
        *,
        trainer: ModelTrainer,
        ignore=(),
    ):
        self._threshold = threshold
        self.model: ModelHandler = trainer(
            reference_collection.choir,
            reference_collection.keyphrase_votes()
            + reference_collection.relation_votes(False),
            ignore=ignore,
        )

    def __call__(self, annotation_votes, tasks):
        for _, (model, _, anns, labels, features) in self.model(
            annotation_votes
        ).items():
            predictions = model.predict(features)

            assert len(predictions) == len(anns)
            for ann, label, pred in tqdm(
                zip(anns, labels, predictions), total=len(anns)
            ):
                yield ann, (label if pred >= self._threshold else None)


class PredictiveEnsembler(Ensembler):
    def __init__(self, choir: EnsembleChoir, orchestrator, predictor: Predictor):
        super().__init__(choir, orchestrator)
        self._predictor = predictor

    def _do_ensemble(self, to_ensemble: EnsembledCollection):
        self._do_prediction(to_ensemble.keyphrase_votes(), "A")
        self._do_prediction(to_ensemble.relation_votes(), "B")

    def _do_prediction(self, annotation_votes, task):
        for ann, label in self._predictor(annotation_votes, task):
            ann.label = label


class IsolatedPredictor(Predictor):
    def __init__(
        self, taskA_predictor: TrainedPredictor, taskB_predictor: TrainedPredictor,
    ):
        self._taskA_predictor = taskA_predictor
        self._taskB_predictor = taskB_predictor

    def __call__(self, annotation_votes, task):
        if task == "A":
            yield from self._taskA_predictor(annotation_votes, task)
        elif task == "B":
            yield from self._taskB_predictor(annotation_votes, task)
        else:
            raise ValueError("Unknown task!")


def get_trained_predictor(
    reference: EnsembledCollection,
    model_type,
    *,
    mode: Literal["category", "all", "each"],
    ignore=(),
    weighting_table=None,
    **kargs,
):
    handler = model_handler_assistant(
        voters=reference.choir.submissions.keys(),
        model_init=lambda: model_type(random_state=0, **kargs),
        mode=mode,
        weighting_table=weighting_table,
    )

    return TrainedPredictor(
        reference, 0.5, trainer=ModelTrainer(handler), ignore=ignore
    )
