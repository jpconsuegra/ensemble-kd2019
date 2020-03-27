from collections import defaultdict
from pathlib import Path

from matplotlib import pyplot as plt

from scripts.ensemble import EnsembleChoir, EnsembledCollection, EnsembleOrchestrator
from scripts.ensemble.utils import keep_non_annotated_sentences
from scripts.utils import Collection, CollectionV1Handler, CollectionV2Handler


def sort_sentences(collection: EnsembledCollection, *, reverse=False):
    sentences = []
    votes_per_sentence = defaultdict(list)

    for sid, _, votes_per_label in collection.keyphrase_votes():
        votes_per_sentence[sid].append(votes_per_label)
    for sid, _, votes_per_label in collection.relation_votes(False):
        votes_per_sentence[sid].append(votes_per_label)

    for sid, sentence in enumerate(collection.sentences):
        votes = votes_per_sentence[sid]
        score = score_annotations(votes)
        sentences.append((sid, sentence, score))

    sentences.sort(key=lambda x: x[-1], reverse=reverse)
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
    return [
        (sid, sentence, score / len(choir.submissions))
        for sid, sentence, score in sentences
    ]


def plot(sentences, name, **kargs):
    scores = [score for _, _, score in sentences]
    plt.hist(scores, cumulative=True, **kargs)
    plt.hist(scores, **kargs)
    plt.savefig(f"{name}.pdf")
    plt.close()


def performance_per_agreement(ensembled: EnsembledCollection, *, normalized=False):
    gold = Collection()
    collection = Collection()

    ordered = sort_sentences(ensembled, reverse=True)
    ordered = normalize(ordered, ensembled.choir) if normalized else ordered

    yield dict(top=0, score=1, f1=1)
    for i, (sid, _, score) in enumerate(ordered, 1):
        gold_sent = ensembled.choir.gold.sentences[sid]
        ensembled_sent = ensembled.collection.sentences[sid]
        gold.sentences.append(gold_sent)
        collection.sentences.append(ensembled_sent)
        yield dict(top=i, score=score, f1=EnsembleChoir.evaluate(collection, gold))


if __name__ == "__main__":
    ps = Path("./data/ehealth2019/submissions/all")
    pg = Path("./data/ehealth2019/testing")

    choir = EnsembleChoir().load(CollectionV1Handler, ps, pg, best=True)
    choir = keep_non_annotated_sentences(choir)

    # should be equal with binary=False
    orchestrator = EnsembleOrchestrator(binary=True)

    collection = orchestrator(choir)

    sentences = sort_sentences(collection)
    normalized = normalize(sentences, choir)

    for sid, s, score in normalized:
        print(f"{sid:4}\t{s.text}\t{score}")

    plot(sentences, "hist")
    plot(normalized, "hist-norm")

    for item in performance_per_agreement(collection):
        print(item)
