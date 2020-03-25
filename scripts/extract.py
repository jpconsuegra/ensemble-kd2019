from collections import defaultdict
from pathlib import Path

from matplotlib import pyplot as plt

from scripts.ensemble import EnsembleChoir, EnsembledCollection, EnsembleOrchestrator
from scripts.ensemble.utils import keep_non_annotated_sentences
from scripts.utils import Collection, CollectionV1Handler, CollectionV2Handler


def sort_sentences(collection: EnsembledCollection):
    sentences = []
    votes_per_sentence = defaultdict(list)

    for sid, _, votes_per_label in collection.keyphrase_votes():
        votes_per_sentence[sid].append(votes_per_label)
    for sid, _, votes_per_label in collection.relation_votes(False):
        votes_per_sentence[sid].append(votes_per_label)

    for sid, sentence in enumerate(collection.sentences):
        votes = votes_per_sentence[sid]
        score = score_annotations(votes)
        sentences.append((sentence, score))

    sentences.sort(key=lambda x: x[1])
    return sentences


def score_annotations(votes):
    accum = sum(
        len(votes_per_label)
        for votes_per_ann in votes
        for votes_per_label in votes_per_ann.values()
    )
    n_anns = sum(len(votes_per_ann) for votes_per_ann in votes)
    return accum / n_anns


def normalize(sentences, choir):
    return [(sentence, score / len(choir.submissions)) for sentence, score in sentences]


def plot(sentences, name, **kargs):
    scores = [score for _, score in sentences]
    plt.hist(scores, cumulative=True, **kargs)
    plt.hist(scores, **kargs)
    plt.savefig(f"{name}.pdf")
    plt.close()


if __name__ == "__main__":
    ps = Path("./data/submissions/all")
    pg = Path("./data/ehealth2019-testing")

    choir = EnsembleChoir().load(CollectionV1Handler, ps, pg, best=True)
    choir = keep_non_annotated_sentences(choir)

    # should be equal with binary=False
    orchestrator = EnsembleOrchestrator(binary=True)

    collection = orchestrator(choir)

    sentences = sort_sentences(collection)
    normalized = normalize(sentences, choir)

    for s, score in normalized:
        print(f"{s.text}\t{score}")

    plot(sentences, "hist")
    plot(normalized, "hist-norm")
