from pathlib import Path
from collections import defaultdict
from scripts.main import BinaryEnsemble


def sort_sentences(ensemble: BinaryEnsemble):
    sentences = []

    votes_per_sentence = defaultdict(list)
    for (sid, *_), (_, votes) in ensemble.keyphrases.items():
        votes_per_sentence[sid].append(votes)
    for (sid, *_), (_, votes) in ensemble.relations.items():
        votes_per_sentence[sid].append(votes)

    for sid, sentence in enumerate(ensemble.collection.sentences):
        votes = votes_per_sentence[sid]
        score = score_annotations(votes)
        sentences.append((sentence, score))

    sentences.sort(key=lambda x: x[1])
    return sentences


def score_annotations(votes):
    total = sum(len(vote) for vote in votes)
    n_anns = len(votes)
    return total / n_anns


if __name__ == "__main__":
    e = BinaryEnsemble()
    ps = Path("./data/submissions/all")
    pg = Path("./data/testing")
    e.load(ps, pg, best=True)
    e.build()
    sentences = sort_sentences(e)
    for s, score in sentences:
        print(f"{s.text}\t{score}")
