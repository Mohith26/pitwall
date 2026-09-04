"""Grouping thousands of failure logs into the handful of bugs behind them.

A nightly regression run produces failures by the thousand and root causes by
the dozen. The expensive part of triage is not fixing bugs, it is working out
that these 340 failures are the same bug wearing 340 different sets of pointer
values, timestamps, and bench identifiers.

Pipeline:

  normalise   strip everything that varies run to run and carries no
              diagnostic information: addresses, timestamps, durations, ids,
              path prefixes, bench names
  shingle     token k-grams, so word order survives but line order does not
              have to be identical
  minhash     fixed size signature per log, so similarity is a signature
              comparison instead of a set intersection
  band        locality sensitive hashing, so only plausible pairs get compared
              at all instead of all n^2 of them
  union find  transitive closure over the surviving pairs

Normalisation is the step that matters, and it is easy to underrate. Without
it every log is unique and the whole pipeline degenerates into one cluster per
failure, which is the same as having no triage. `experiments.py` measures that
directly rather than asserting it.
"""

import hashlib
import re
from collections import defaultdict

_SUBSTITUTIONS = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b"), "<ts>"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<uuid>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<addr>"),
    (re.compile(r"\b[0-9a-fA-F]{12,}\b"), "<hash>"),
    (re.compile(r"\bhil-\d+\b|\bsim-\d+\b"), "<bench>"),
    (re.compile(r"\brun[_-]?\d+\b", re.I), "<run>"),
    (re.compile(r"\b(?:thread|tid|pid)[ =:_-]*\d+\b", re.I), "<tid>"),
    (re.compile(r"\b\d+(?:\.\d+)?(?:ms|us|s|Hz|MB|KB|GB)\b"), "<qty>"),
    (re.compile(r"(?:/[\w.\-]+)+/([\w.\-]+)"), r"\1"),
    (re.compile(r"\b\d+\.\d+\b"), "<float>"),
    (re.compile(r"\b\d+\b"), "<int>"),
    (re.compile(r"\s+"), " "),
]


def normalize(log):
    """Collapse everything that changes between two runs of the same bug."""
    text = log
    for pattern, replacement in _SUBSTITUTIONS:
        text = pattern.sub(replacement, text)
    return text.strip().lower()


def shingles(text, k=3):
    tokens = text.split()
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


_MASK = (1 << 61) - 1


def _hash(value, seed):
    digest = hashlib.blake2b(
        value.encode("utf-8"), digest_size=8, salt=seed.to_bytes(8, "big")[:8]
    ).digest()
    return int.from_bytes(digest, "big") & _MASK


def minhash(shingle_set, num_perm=128):
    """Signature as the minimum hash under each of num_perm hash functions.

    The probability that two signatures agree in a given position equals the
    Jaccard similarity of the underlying sets, which is why this works, and it
    is also the property `experiments.py` checks rather than trusts.
    """
    if not shingle_set:
        return tuple([_MASK] * num_perm)
    sig = [_MASK] * num_perm
    for shingle in shingle_set:
        for i in range(num_perm):
            h = _hash(shingle, i)
            if h < sig[i]:
                sig[i] = h
    return tuple(sig)


def signature_similarity(sig_a, sig_b):
    same = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return same / float(len(sig_a))


def band_candidates(signatures, bands, rows):
    """Locality sensitive hashing: two logs become candidates if any band of
    `rows` signature positions matches exactly.

    The threshold is approximately (1/bands) ** (1/rows), and the curve is a
    step function steep enough that pairs well below it almost never surface.
    That is the point: it turns an n^2 comparison into a bucket lookup, at the
    cost of missing some true pairs near the threshold.
    """
    if bands * rows > len(signatures[0]):
        raise ValueError("bands * rows exceeds the signature length")
    buckets = defaultdict(list)
    for idx, sig in enumerate(signatures):
        for b in range(bands):
            key = (b, sig[b * rows:(b + 1) * rows])
            buckets[key].append(idx)
    pairs = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
    return pairs


def lsh_threshold(bands, rows):
    return (1.0 / bands) ** (1.0 / rows)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster(logs, threshold=0.45, num_perm=128, bands=32, rows=4, k=3, use_lsh=True,
            do_normalize=True, require_support=True):
    """Cluster raw failure logs. Returns a label per log.

    `require_support` is here because plain transitive closure over a
    similarity graph is fragile in a way that is easy to miss. Two genuinely
    different bugs need only one log that resembles both to be welded into a
    single cluster, and union find will happily do it. On one seed at a 0.45
    threshold that produced a single cluster of 133 logs containing four
    distinct root causes, while the overall cluster count still looked
    plausible, which is the worst kind of wrong: it does not look broken.

    With support required, an edge only merges if some third log is also
    similar to both endpoints. A single bridging log has no triangle to stand
    on and the merge is refused. Real clusters are dense and have triangles
    everywhere, so they are unaffected.

    Labels are renumbered by first appearance, so two runs over the same input
    produce the same labels.
    """
    prepared = [shingles(normalize(l) if do_normalize else l, k) for l in logs]
    sigs = [minhash(s, num_perm) for s in prepared]
    if use_lsh:
        candidates = band_candidates(sigs, bands, rows)
    else:
        candidates = {
            (i, j) for i in range(len(logs)) for j in range(i + 1, len(logs))
        }
    uf = UnionFind(len(logs))
    compared = 0
    edges = []
    neighbours = defaultdict(set)
    for i, j in candidates:
        compared += 1
        if signature_similarity(sigs[i], sigs[j]) >= threshold:
            edges.append((i, j))
            neighbours[i].add(j)
            neighbours[j].add(i)
    merged = 0
    refused = 0
    for i, j in edges:
        if require_support and not (neighbours[i] & neighbours[j]):
            refused += 1
            continue
        merged += 1
        uf.union(i, j)
    roots = {}
    labels = []
    for i in range(len(logs)):
        r = uf.find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels.append(roots[r])
    return labels, {"pairs_compared": compared,
                    "possible_pairs": len(logs) * (len(logs) - 1) // 2,
                    "edges_above_threshold": len(edges),
                    "edges_merged": merged,
                    "edges_refused_for_lack_of_support": refused}


def pairwise_scores(predicted, truth):
    """Precision, recall, and F1 over pairs of logs placed together.

    Pairwise rather than cluster counting, because the two ways of being wrong
    matter differently. Splitting one bug across three clusters wastes three
    engineers' mornings. Merging three bugs into one cluster means two of them
    get closed as duplicates of the third and ship.
    """
    n = len(predicted)
    tp = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_pred = predicted[i] == predicted[j]
            same_true = truth[i] == truth[j]
            if same_pred and same_true:
                tp += 1
            elif same_pred and not same_true:
                fp += 1
            elif not same_pred and same_true:
                fn += 1
    precision = tp / float(tp + fp) if tp + fp else 1.0
    recall = tp / float(tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "clusters_predicted": len(set(predicted)),
        "clusters_true": len(set(truth)),
    }
