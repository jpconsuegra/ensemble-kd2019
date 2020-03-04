from collections import defaultdict
from typing import Any, Dict, List, Literal, Tuple

import numpy as np

from scripts.utils import ENTITIES, RELATIONS


class FeatureBuilder:
    def __call__(self, raw_features):
        raise NotImplementedError()


class VotingFeatures(FeatureBuilder):
    def __init__(self, voters):
        self._voters = voters

    def __call__(self, votes):
        n_votes = len(self._voters)
        features = np.zeros(n_votes)
        voted = [i for i, submit in enumerate(self._voters) if submit in votes]
        features[voted] = 1
        return features


class LabelFeatures(FeatureBuilder):
    def __init__(self, labels):
        self._label2index = {label: i for i, label in enumerate(labels)}

    def __call__(self, label):
        features = np.zeros(len(self._label2index))
        index = self._label2index[label]
        features[index] = 1
        return features


class WithHandler(FeatureBuilder):
    def __init__(self, builder: FeatureBuilder, handler):
        self._builder = builder
        self._handler = handler

    def __call__(self, raw_features):
        return self._builder(self._handler(raw_features))


class ConcatenatedFeatures(FeatureBuilder):
    def __init__(self, *builders_and_handlers):
        self._builders = [
            WithHandler(builder=b, handler=h) for b, h in builders_and_handlers
        ]

    def __call__(self, raw_features):
        return np.concatenate([builder(raw_features) for builder in self._builders])


class ModelHandler:
    def __call__(
        self, annotation_votes, selected_sids=None
    ) -> Dict[str, Tuple[Any, List[int], List[Any], List[str], np.ndarray]]:
        pass


class PerCategoryModel(ModelHandler):
    def __init__(self, *, voters, labels_per_category: Dict[str, list], model_init):
        self._builders = {
            category: self._get_builder_according_to_labels(labels, voters)
            for category, labels in labels_per_category.items()
        }
        self._models = {category: model_init() for category in labels_per_category}
        self._label2category = {
            label: category
            for category, labels in labels_per_category.items()
            for label in labels
        }
        assert sum(len(x) for x in labels_per_category.values()) == len(
            self._label2category
        )

    @classmethod
    def _get_builder_according_to_labels(cls, labels, voters):
        if len(labels) > 1:
            return ConcatenatedFeatures(
                (LabelFeatures(labels), cls._get_label),
                (VotingFeatures(voters), cls._get_votes),
            )
        else:
            return WithHandler(VotingFeatures(voters), cls._get_votes)

    @classmethod
    def _get_label(cls, item):
        label, _ = item
        return label

    @classmethod
    def _get_votes(cls, item):
        _, votes = item
        return votes

    def __call__(self, annotation_votes, selected_sids=None):
        per_category = defaultdict(lambda: ([], [], [], []))
        for sid, ann, votes_per_label in annotation_votes:
            if selected_sids is None or sid in selected_sids:
                # TODO: en el caso no binario estoy no hace lo esperado
                for label, votes in votes_per_label.items():
                    if label not in self._label2category:
                        print(f"Ignoring {ann} with label {label}.")
                        continue

                    category = self._label2category[label]
                    builder = self._builders[category]
                    sids, anns, labels, features = per_category[category]

                    features.append(builder((label, votes)))
                    sids.append(sid)
                    anns.append(ann)
                    labels.append(label)

        return {
            category: (self._models[category], sids, anns, labels, np.asarray(features))
            for category, (sids, anns, labels, features) in per_category.items()
        }


class AllInOneModel(PerCategoryModel):
    def __init__(self, *, voters, labels, model_init):
        super().__init__(
            voters=voters, labels_per_category={"all": labels}, model_init=model_init
        )


class PerLabelModel(PerCategoryModel):
    def __init__(self, *, voters, labels, model_init):
        super().__init__(
            voters=voters,
            labels_per_category={l: [l] for l in labels},
            model_init=model_init,
        )


def model_handler_assistant(
    *,
    voters,
    model_init,
    labels=ENTITIES + RELATIONS,
    mode: Literal["category", "all", "each"],
):
    if mode == "category":
        labels_per_category = defaultdict(list)
        for label in labels:
            if label in ENTITIES:
                labels_per_category["ENTITIES"].append(label)
            elif label in RELATIONS:
                labels_per_category["RELATIONS"].append(label)
            else:
                raise Exception("Unknown category!")
        return lambda: PerCategoryModel(
            voters=voters,
            labels_per_category=labels_per_category,
            model_init=model_init,
        )
    elif mode == "all":
        return lambda: AllInOneModel(
            voters=voters, labels=labels, model_init=model_init
        )
    elif mode == "each":
        return lambda: PerLabelModel(
            voters=voters, labels=labels, model_init=model_init
        )
    else:
        raise ValueError("Unknown mode: {mode}")
