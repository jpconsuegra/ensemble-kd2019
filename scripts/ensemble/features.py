from collections import defaultdict
import numpy as np


class VotingFeatures:
    def __init__(self, voters):
        self._voters = voters

    def __call__(self, votes):
        n_votes = len(self._voters)
        features = np.zeros(n_votes)
        voted = [i for i, submit in enumerate(self._voters) if submit in votes]
        features[voted] = 1
        return features


class LabelFeatures:
    def __init__(self, labels):
        self._labels = labels

    def __call__(self, label):
        index = self._labels.index(label)
        features = np.zeros(len(self._labels))
        features[index] = 1
        return features


class ConcatenatedFeatures:
    def __init__(self, *builders_and_handlers):
        self._builders_and_handlers = builders_and_handlers

    def __call__(self, item):
        return np.concatenate(
            [builder(handler(item)) for builder, handler in self._builders_and_handlers]
        )


class AllInOneModel:
    def __init__(self, *, voters, labels, model_init):
        self._builder = ConcatenatedFeatures(
            (LabelFeatures(labels), self._get_label),
            (VotingFeatures(voters), self._get_votes),
        )
        self._model = model_init()

    def _get_label(self, item):
        ann, _ = item
        return ann.label

    def _get_votes(self, item):
        _, votes = item
        return votes

    def __call__(self, annotations, selected_sids=None):
        sids = []
        anns = []
        features = []
        for (sid, *_), (ann, votes) in annotations.items():
            if selected_sids is None or sid in selected_sids:
                features.append(self._builder((ann, votes)))
                sids.append(sid)
                anns.append(ann)
        return [("all", self._model, sids, anns, np.asarray(features))]


class PerLabelModel:
    def __init__(self, *, voters, labels, model_init):
        self._builder = VotingFeatures(voters)
        self._models = {label: model_init() for label in labels}

    def __call__(self, annotations, selected_sids=None):
        per_label = defaultdict(lambda: ([], [], []))
        for (sid, *_), (ann, votes) in annotations.items():
            if selected_sids is None or sid in selected_sids:
                sids, anns, features = per_label[ann.label]

                features.append(self._builder(votes))
                sids.append(sid)
                anns.append(ann)

        return [
            (label, self._models[label], sids, anns, np.asarray(features))
            for label, (sids, anns, features) in per_label.items()
        ]
