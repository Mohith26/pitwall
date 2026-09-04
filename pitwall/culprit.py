"""Which commit broke the test, when the test is flaky.

Plain bisection assumes the oracle is deterministic. Run the test at the
midpoint, believe the answer, halve the range. On a flaky test that assumption
is false in the worst possible way: one unlucky pass on a genuinely broken
commit sends the search into the half that contains nothing, and there is no
mechanism by which it ever recovers. Every subsequent step is confidently
searching the wrong place.

Two better options are implemented, and the first one I reached for was the
worse of the two.

`bayesian_bisect` keeps a full posterior over which commit is the culprit and
updates it after every single test run, probing at the posterior median. It is
the obvious way to make bisection Bayesian and it converges slowly, because
moving the probe after one observation throws away almost all of the value of
that observation: a single sample barely moves the posterior, so the next probe
is usually the same place anyway, just reached more expensively.

`sprt_bisect` keeps the classical structure and fixes the oracle instead. At
each bisection step it stays on that one commit and accumulates a log
likelihood ratio until Wald's sequential test crosses a threshold, then makes
the decision and permanently halves the range. It uses fewer test runs and is
far more accurate, and it is the standard answer to binary search over a noisy
oracle.

`experiments.py culprit` measures both against naive bisection across flake
rates.
"""

import math
import random


_EPS = 1e-3


def _clamp(p, eps=_EPS):
    return min(1.0 - eps, max(eps, p))


def make_flaky_test(true_culprit, p_pass_good, p_pass_bad, rng):
    """An oracle that passes with probability p_pass_good before the culprit
    commit and p_pass_bad at or after it. Counts its own invocations, because
    accuracy per test run is the number that matters, not accuracy alone."""
    state = {"runs": 0}

    def run(commit):
        state["runs"] += 1
        p = p_pass_good if commit < true_culprit else p_pass_bad
        return rng.random() < p

    return run, state


def naive_bisect(n_commits, run_test):
    lo, hi = 0, n_commits - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if run_test(mid):
            lo = mid + 1
        else:
            hi = mid
    return lo


def repeated_bisect(n_commits, run_test, repeats=5):
    """The pragmatic middle ground people actually ship: run the test a fixed
    number of times at each probe and take the majority. Better than one shot,
    and it spends the same budget at every node whether the evidence is
    ambiguous or overwhelming."""
    lo, hi = 0, n_commits - 1
    while lo < hi:
        mid = (lo + hi) // 2
        passes = sum(1 for _ in range(repeats) if run_test(mid))
        if passes * 2 > repeats:
            lo = mid + 1
        else:
            hi = mid
    return lo


def bayesian_bisect(n_commits, run_test, p_pass_good, p_pass_bad,
                    confidence=0.95, max_queries=400):
    posterior = [1.0 / n_commits] * n_commits
    queries = 0

    def median_index():
        acc = 0.0
        for i, p in enumerate(posterior):
            acc += p
            if acc >= 0.5:
                return i
        return n_commits - 1

    while queries < max_queries:
        best = max(range(n_commits), key=lambda i: posterior[i])
        if posterior[best] >= confidence:
            return best
        probe = median_index()
        result = run_test(probe)
        queries += 1
        total = 0.0
        updated = [0.0] * n_commits
        for k in range(n_commits):
            # commit `probe` is broken exactly when the culprit is at or before it
            p_pass = _clamp(p_pass_bad) if probe >= k else _clamp(p_pass_good)
            likelihood = p_pass if result else (1.0 - p_pass)
            updated[k] = posterior[k] * likelihood
            total += updated[k]
        if total <= 0:
            break
        posterior = [p / total for p in updated]
    return max(range(n_commits), key=lambda i: posterior[i])


def sprt_bisect(n_commits, run_test, p_pass_good, p_pass_bad,
                node_confidence=0.99, max_samples_per_node=80):
    """Noisy binary search. Wald's sequential probability ratio test at each
    probe, then a normal bisection step.

    The sequential part is what makes it cheap: an unambiguous probe crosses a
    threshold in two or three runs, and only the genuinely marginal ones near
    the culprit spend the full budget.
    """
    # Clamp the rates away from 0 and 1. A deterministic test gives
    # p_pass_bad = 0, and log(0) is not a number the test can carry. More to
    # the point, an infinite log likelihood ratio means a single observation is
    # treated as absolute proof, so one mislabelled or genuinely impossible
    # outcome pins the decision forever with no way back. In practice these
    # rates are estimated from history anyway and an estimate of exactly 0 is
    # an artefact of a small sample, not a fact about the test.
    p_good = _clamp(p_pass_good)
    p_bad = _clamp(p_pass_bad)
    alpha = 1.0 - node_confidence
    upper = math.log((1.0 - alpha) / alpha)
    lower = math.log(alpha / (1.0 - alpha))
    llr_pass = math.log(p_bad / p_good)
    llr_fail = math.log((1.0 - p_bad) / (1.0 - p_good))
    lo, hi = 0, n_commits - 1
    while lo < hi:
        mid = (lo + hi) // 2
        llr = 0.0
        decision = None
        for _ in range(max_samples_per_node):
            llr += llr_pass if run_test(mid) else llr_fail
            if llr >= upper:
                decision = "bad"
                break
            if llr <= lower:
                decision = "good"
                break
        if decision is None:
            decision = "bad" if llr > 0 else "good"
        if decision == "bad":
            hi = mid
        else:
            lo = mid + 1
    return lo


def evaluate(strategy, n_commits, p_pass_good, p_pass_bad, trials, seed, **kw):
    """Accuracy, cost, and how far off it was when it was wrong.

    Distance matters separately from accuracy. A culprit finder that is usually
    exact and occasionally off by one is a useful tool. One that is usually
    exact and occasionally off by two hundred commits is a coin flip with extra
    steps, and plain accuracy does not distinguish them.
    """
    rng = random.Random(seed)
    exact = 0
    within_one = 0
    total_runs = 0
    distances = []
    for _ in range(trials):
        truth = rng.randrange(1, n_commits)
        run, state = make_flaky_test(truth, p_pass_good, p_pass_bad, rng)
        if strategy in (bayesian_bisect, sprt_bisect):
            found = strategy(n_commits, run, p_pass_good, p_pass_bad, **kw)
        else:
            found = strategy(n_commits, run, **kw)
        total_runs += state["runs"]
        d = abs(found - truth)
        distances.append(d)
        if d == 0:
            exact += 1
        if d <= 1:
            within_one += 1
    distances.sort()
    return {
        "strategy": strategy.__name__,
        "accuracy": exact / float(trials),
        "within_one_commit": within_one / float(trials),
        "mean_test_runs": total_runs / float(trials),
        "median_distance": distances[len(distances) // 2],
        "p95_distance": distances[int(0.95 * len(distances))],
        "trials": trials,
    }
