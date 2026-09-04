"""Deciding whether a build is good enough to qualify, and which tests are too
unreliable to be part of that decision.

Two things this refuses to do.

It will not gate on a raw pass rate. 47 of 50 is 94%, and so is 470 of 500, and
they are not the same evidence. The gate uses the lower bound of a Wilson score
interval, so a build that has barely been tested cannot pass by being lucky.

It will not quarantine a test just because it fails a lot. A test that fails
every single time is not flaky, it is a regression, and quarantining it is how
a real bug ships. Quarantine requires a test to have shown both outcomes.
"""

import math

_Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


def wilson_interval(successes, n, confidence=0.95):
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves at small n and extreme rates, which is precisely the regime a
    release gate operates in: the interesting builds are the ones with few runs
    or a pass rate near 1.
    """
    if n == 0:
        return 0.0, 1.0
    z = _Z.get(confidence, 1.96)
    p = successes / float(n)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    margin = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def wilson_lower_bound(successes, n, confidence=0.95):
    return wilson_interval(successes, n, confidence)[0]


def runs_needed_for(required_lower_bound, confidence=0.95, max_n=100000):
    """How many consecutive passes it takes to clear a bar.

    Useful for arguing about gate settings with a number instead of an opinion:
    a 0.99 lower bound with everything passing needs a specific run count, and
    if the fleet cannot supply it, the gate is decoration.
    """
    for n in range(1, max_n):
        if wilson_lower_bound(n, n, confidence) >= required_lower_bound:
            return n
    return None


def estimate_flake_rate(history):
    if not history:
        return None
    return 1.0 - sum(1 for x in history if x) / float(len(history))


def classify(history, min_runs=10, flake_threshold=0.10):
    """One of stable, flaky, broken, or unknown.

    The distinction between flaky and broken is the entire point. A test with
    a 30% failure rate that sometimes passes is flaky and should stop blocking
    releases while somebody fixes it. A test with a 100% failure rate is broken
    and must keep blocking, and treating the two the same is how a quarantine
    list becomes a place bugs go to hide.
    """
    if len(history) < min_runs:
        return "unknown"
    failures = sum(1 for x in history if not x)
    if failures == 0:
        return "stable"
    if failures == len(history):
        return "broken"
    rate = failures / float(len(history))
    return "flaky" if rate > flake_threshold else "stable"


def quarantine_set(histories, min_runs=10, flake_threshold=0.10):
    quarantined = set()
    classes = {}
    for name, history in histories.items():
        verdict = classify(history, min_runs, flake_threshold)
        classes[name] = verdict
        if verdict == "flaky":
            quarantined.add(name)
    return quarantined, classes


def qualify(results, histories, required_lower_bound=0.95, confidence=0.95,
            unschedulable=0):
    """Gate a build.

    `results` maps test name to pass/fail for this build. Quarantined tests are
    excluded from the pass rate but reported. Unschedulable cases block
    outright: a suite where some cases could not run anywhere did not actually
    run, and computing a pass rate over the subset that did is how a green
    dashboard hides a fleet configuration bug.
    """
    quarantined, classes = quarantine_set(histories)
    counted = {k: v for k, v in results.items() if k not in quarantined}
    broken = sorted(k for k, v in classes.items() if v == "broken")
    successes = sum(1 for v in counted.values() if v)
    n = len(counted)
    lower, upper = wilson_interval(successes, n, confidence)
    reasons = []
    if unschedulable:
        reasons.append("%d cases had no capable bench" % unschedulable)
    if broken:
        reasons.append("%d consistently failing tests: %s"
                       % (len(broken), ", ".join(broken[:5])))
    if lower < required_lower_bound:
        reasons.append("pass rate lower bound %.4f is under the %.2f bar"
                       % (lower, required_lower_bound))
    return {
        "qualified": not reasons,
        "reasons": reasons,
        "counted_tests": n,
        "passed": successes,
        "raw_pass_rate": successes / float(n) if n else 0.0,
        "wilson_lower": lower,
        "wilson_upper": upper,
        "quarantined": sorted(quarantined),
        "broken": broken,
        "unschedulable": unschedulable,
    }
