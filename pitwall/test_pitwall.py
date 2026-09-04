"""Test suite. `python -m pitwall.test_pitwall`.

Where a closed form or an exhaustive answer exists, that is what the code is
checked against: brute force optimal makespan for the scheduler, a published
worked Wilson interval, the Jaccard identity that MinHash is supposed to
estimate, and ground truth labels for clustering and culprit finding.
"""

import random
import sys
import time

from . import triage
from .culprit import (
    bayesian_bisect,
    evaluate,
    make_flaky_test,
    naive_bisect,
    repeated_bisect,
    sprt_bisect,
)
from .harness import make_failure_corpus, make_failure_log, make_suite, ROOT_CAUSES
from .qualification import (
    classify,
    qualify,
    quarantine_set,
    runs_needed_for,
    wilson_interval,
    wilson_lower_bound,
)
from .scheduler import (
    lower_bound,
    optimal_makespan,
    schedule,
    utilization,
)
from .spec import (
    ALL_CAPABILITIES,
    Bench,
    default_fleet,
    expand,
    unschedulable,
    validate_suite,
)

_T0 = time.time()
R = {"passed": 0, "assertions": 0, "failed": 0, "failures": []}


def check(name, cond, detail=""):
    R["assertions"] += 1
    if not cond:
        R["failed"] += 1
        R["failures"].append("%s: %s" % (name, detail))
        print("  FAIL %s %s" % (name, detail))
    return cond


def case(fn):
    before = R["failed"]
    fn()
    if R["failed"] == before:
        R["passed"] += 1
        print("  ok   %s" % fn.__name__)
    return fn


# -------------------------------------------------------------------- spec


@case
def test_validation_catches_the_silent_failures():
    bad = {"cases": [
        {"name": "Bad Name", "requires": [], "priority": 0, "duration": 1},
        {"name": "typo_cap", "requires": ["camrea"], "priority": 0, "duration": 1},
        {"name": "dup", "requires": [], "priority": 0, "duration": 1},
        {"name": "dup", "requires": [], "priority": 0, "duration": 1},
        {"name": "zero_time", "requires": [], "priority": 0, "duration": 0},
        {"name": "neg_prio", "requires": [], "priority": -1, "duration": 1},
        {"name": "empty_axis", "requires": [], "priority": 0, "duration": 1,
         "axes": {"weather": []}},
        {"name": "too_many_retries", "requires": [], "priority": 0, "duration": 1,
         "retries": 9},
    ]}
    problems = validate_suite(bad, ALL_CAPABILITIES)
    joined = " | ".join(problems)
    for expected in ["name must match", "unknown capability", "duplicate case name",
                     "duration must be positive", "priority must be",
                     "deletes every expansion", "retries must be"]:
        check("validation reports %r" % expected, expected in joined, joined)
    check("empty suite is rejected", validate_suite({"cases": []}, ALL_CAPABILITIES))


@case
def test_good_suite_validates_clean():
    suite = make_suite(50, seed=1)
    check("no problems", validate_suite(suite, ALL_CAPABILITIES) == [],
          str(validate_suite(suite, ALL_CAPABILITIES)))


@case
def test_expansion_is_deterministic_and_complete():
    suite = {"cases": [{
        "name": "night_merge", "requires": ["simulation"], "priority": 1, "duration": 4,
        "axes": {"weather": ["rain", "clear"], "speed": ["30", "60", "90"]},
    }]}
    a = [c.name for c in expand(suite)]
    b = [c.name for c in expand(suite)]
    check("6 expansions", len(a) == 6, str(a))
    check("stable ordering", a == b)
    check("sorted", a == sorted(a), str(a))
    check("axes recorded", expand(suite)[0].axes == {"speed": "30", "weather": "clear"},
          str(expand(suite)[0].axes))


@case
def test_unschedulable_cases_are_found_not_hidden():
    benches = [Bench("only_sim", ["simulation"])]
    cases = expand(make_suite(200, seed=2))
    orphans = unschedulable(cases, benches)
    check("a sim only fleet cannot host the hardware cases", len(orphans) > 50,
          str(len(orphans)))
    check("none of the orphans need only simulation",
          all(c.requires != frozenset(["simulation"]) for c in orphans))


# --------------------------------------------------------------- scheduler


@case
def test_schedule_respects_capabilities_and_exclusivity():
    benches = default_fleet()
    cases = expand(make_suite(300, seed=3))
    assignments, skipped, makespan = schedule(cases, benches)
    check("nothing skipped on the full fleet", skipped == [], str(len(skipped)))
    by_id = {b.bench_id: b for b in benches}
    for a in assignments:
        check("%s is capable of %s" % (a.bench_id, a.case.name),
              a.case.requires <= by_id[a.bench_id].capabilities)
    per_bench = {}
    for a in assignments:
        per_bench.setdefault(a.bench_id, []).append((a.start, a.end))
    overlaps = 0
    for spans in per_bench.values():
        spans.sort()
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            if s2 < e1 - 1e-9:
                overlaps += 1
    check("no bench runs two cases at once", overlaps == 0, str(overlaps))
    check("makespan is the last finish",
          abs(makespan - max(a.end for a in assignments)) < 1e-9)


@case
def test_schedule_beats_the_lower_bound_but_not_by_magic():
    benches = default_fleet()
    cases = expand(make_suite(400, seed=4))
    _, _, makespan = schedule(cases, benches)
    lb = lower_bound(cases, benches)
    check("makespan cannot beat the lower bound", makespan >= lb - 1e-9,
          "%.2f vs %.2f" % (makespan, lb))
    check("and is within a small factor of it", makespan < lb * 2.0,
          "%.2f vs %.2f" % (makespan, lb))
    print("       400 cases on 8 benches: makespan %.1f, lower bound %.1f (%.2fx)"
          % (makespan, lb, makespan / lb))


@case
def test_scheduler_against_brute_force_optimal():
    rng = random.Random(5)
    ratios = []
    solved = 0
    for _ in range(40):
        benches = default_fleet()[:3]
        suite = make_suite(7, seed=rng.randrange(10000))
        cases = [c for c in expand(suite)]
        for c in cases:
            c.requires = frozenset()
        opt = optimal_makespan(cases, benches, node_budget=200000)
        if opt is None:
            continue
        solved += 1
        _, _, got = schedule(cases, benches)
        check("heuristic never beats optimal", got >= opt - 1e-9,
              "%.3f vs %.3f" % (got, opt))
        ratios.append(got / opt)
    check("solved enough instances", solved > 20, str(solved))
    worst = max(ratios)
    mean = sum(ratios) / len(ratios)
    check("stays inside the 2 - 1/m list scheduling factor", worst <= 2.0 - 1.0 / 3,
          "worst %.3f" % worst)
    print("       vs brute force optimal on %d instances: mean %.3fx, worst %.3fx"
          % (solved, mean, worst))


@case
def test_utilization_is_sane():
    benches = default_fleet()
    cases = expand(make_suite(500, seed=6))
    assignments, _, makespan = schedule(cases, benches)
    u = utilization(assignments, benches, makespan)
    check("utilisation in range", 0.0 < u <= 1.0, "%.3f" % u)


# ------------------------------------------------------------------ triage


@case
def test_normalisation_removes_only_the_noise():
    log = ("2027-03-04T11:22:33.456Z E replay: checksum mismatch, expected 0xdeadbeef "
           "got 0xfeedface on hil-03 run_884412 tid=4471 after 1240ms")
    norm = triage.normalize(log)
    for gone in ["2027-03-04", "0xdeadbeef", "hil-03", "884412", "4471"]:
        check("%r removed" % gone, gone not in norm, norm)
    for kept in ["replay", "checksum", "mismatch", "expected"]:
        check("%r kept" % kept, kept in norm, norm)
    rng = random.Random(7)
    a = triage.normalize(make_failure_log(0, rng))
    b = triage.normalize(make_failure_log(0, rng))
    check("two logs from one cause become similar",
          triage.jaccard(triage.shingles(a), triage.shingles(b)) > 0.25,
          "%.3f" % triage.jaccard(triage.shingles(a), triage.shingles(b)))


@case
def test_minhash_estimates_jaccard():
    """The whole method rests on P(signatures agree at a position) equalling the
    Jaccard similarity, so this checks that rather than assuming it."""
    rng = random.Random(11)
    worst = 0.0
    for _ in range(25):
        universe = ["tok%d" % i for i in range(200)]
        a = set(rng.sample(universe, 80))
        b = set(rng.sample(universe, 80))
        true_j = triage.jaccard(a, b)
        est = triage.signature_similarity(triage.minhash(a, 256), triage.minhash(b, 256))
        worst = max(worst, abs(est - true_j))
    check("MinHash tracks Jaccard within sampling error", worst < 0.08, "%.4f" % worst)


@case
def test_clustering_recovers_the_root_causes():
    logs, truth = make_failure_corpus(220, seed=13)
    labels, stats = triage.cluster(logs)
    scores = triage.pairwise_scores(labels, truth)
    check("pairwise F1 is high", scores["f1"] > 0.80, str(scores))
    check("precision matters more than recall and is high",
          scores["precision"] > 0.90, str(scores))
    check("LSH avoided most pairs",
          stats["pairs_compared"] < stats["possible_pairs"] * 0.5, str(stats))
    print("       220 logs, %d causes: F1 %.3f, precision %.3f, recall %.3f, "
          "%d clusters, %d pairs compared of %d"
          % (scores["clusters_true"], scores["f1"], scores["precision"],
             scores["recall"], scores["clusters_predicted"],
             stats["pairs_compared"], stats["possible_pairs"]))


@case
def test_transitive_closure_welds_unrelated_bugs_together():
    """Plain union find over a similarity graph is one bridging log away from
    merging two different bugs. Requiring a triangle refuses about one percent
    of edges and stops it."""
    from collections import Counter
    logs, truth = make_failure_corpus(200, seed=13)
    naive_labels, naive_stats = triage.cluster(logs, require_support=False)
    supported, sup_stats = triage.cluster(logs, require_support=True)

    def widest(labels):
        counts = Counter(labels)
        biggest = counts.most_common(1)[0][0]
        causes = {truth[i] for i, l in enumerate(labels) if l == biggest}
        return counts[biggest], len(causes)

    naive_size, naive_causes = widest(naive_labels)
    sup_size, sup_causes = widest(supported)
    check("plain closure builds a blob spanning several root causes",
          naive_causes >= 3, "%d logs, %d causes" % (naive_size, naive_causes))
    check("requiring support shrinks it", sup_causes < naive_causes,
          "%d causes vs %d" % (sup_causes, naive_causes))
    refused = sup_stats["edges_refused_for_lack_of_support"]
    total = sup_stats["edges_above_threshold"]
    check("and it refuses only a tiny fraction of edges",
          refused / float(total) < 0.05, "%d of %d" % (refused, total))
    check("naive mode refuses nothing",
          naive_stats["edges_refused_for_lack_of_support"] == 0)
    print("       biggest cluster: %d logs spanning %d causes without support, "
          "%d logs spanning %d with; %d of %d edges refused"
          % (naive_size, naive_causes, sup_size, sup_causes, refused, total))


@case
def test_clustering_collapses_without_normalisation():
    logs, truth = make_failure_corpus(150, seed=17)
    with_norm = triage.pairwise_scores(triage.cluster(logs)[0], truth)
    without = triage.pairwise_scores(triage.cluster(logs, do_normalize=False)[0], truth)
    check("normalisation is what makes this work",
          with_norm["f1"] > without["f1"] + 0.2,
          "with %.3f without %.3f" % (with_norm["f1"], without["f1"]))
    print("       F1 with normalisation %.3f, without %.3f"
          % (with_norm["f1"], without["f1"]))


@case
def test_pairwise_scores_edge_cases():
    check("perfect", triage.pairwise_scores([0, 0, 1, 1], [0, 0, 1, 1])["f1"] == 1.0)
    check("everything merged has recall 1",
          triage.pairwise_scores([0, 0, 0, 0], [0, 0, 1, 1])["recall"] == 1.0)
    check("everything split has precision 1",
          triage.pairwise_scores([0, 1, 2, 3], [0, 0, 1, 1])["precision"] == 1.0)


# ----------------------------------------------------------------- culprit


@case
def test_bisection_is_exact_on_a_deterministic_oracle():
    for n in (2, 7, 64, 129, 1000):
        for truth in (1, n // 3, n - 1):
            rng = random.Random(0)
            run, _ = make_flaky_test(truth, 1.0, 0.0, rng)
            check("naive bisect finds %d of %d" % (truth, n),
                  naive_bisect(n, run) == truth)


@case
def test_sprt_beats_naive_and_bayesian_under_flakiness():
    n = 128
    naive = evaluate(naive_bisect, n, 0.95, 0.30, trials=150, seed=19)
    repeated = evaluate(repeated_bisect, n, 0.95, 0.30, trials=150, seed=19, repeats=5)
    sprt = evaluate(sprt_bisect, n, 0.95, 0.30, trials=150, seed=19)
    check("naive bisection is badly broken here", naive["accuracy"] < 0.4, str(naive))
    check("sprt is much better", sprt["accuracy"] > 0.85, str(sprt))
    check("sprt beats majority voting", sprt["accuracy"] > repeated["accuracy"],
          "sprt %.3f repeated %.3f" % (sprt["accuracy"], repeated["accuracy"]))
    print("       flaky oracle (30%% pass when broken): naive %.2f, majority-of-5 %.2f, sprt %.2f"
          % (naive["accuracy"], repeated["accuracy"], sprt["accuracy"]))
    print("       test runs used: naive %.1f, majority-of-5 %.1f, sprt %.1f"
          % (naive["mean_test_runs"], repeated["mean_test_runs"], sprt["mean_test_runs"]))


@case
def test_bayesian_version_is_the_worse_idea():
    n = 64
    bayes = evaluate(bayesian_bisect, n, 0.95, 0.30, trials=60, seed=23)
    sprt = evaluate(sprt_bisect, n, 0.95, 0.30, trials=60, seed=23)
    check("sprt is at least as accurate", sprt["accuracy"] >= bayes["accuracy"],
          "sprt %.3f bayes %.3f" % (sprt["accuracy"], bayes["accuracy"]))
    print("       posterior-median Bayesian: %.2f accuracy in %.1f runs; sprt %.2f in %.1f"
          % (bayes["accuracy"], bayes["mean_test_runs"],
             sprt["accuracy"], sprt["mean_test_runs"]))


# ----------------------------------------------------------- qualification


@case
def test_wilson_interval_against_a_worked_example():
    lo, hi = wilson_interval(9, 10, 0.95)
    check("lower", abs(lo - 0.5958) < 0.002, "%.4f" % lo)
    check("upper", abs(hi - 0.9821) < 0.002, "%.4f" % hi)
    check("stays in range at the extremes", wilson_interval(0, 5)[0] == 0.0)
    check("upper below 1 for a finite sample", wilson_interval(5, 5)[1] <= 1.0)
    check("more runs at the same rate give a tighter bound",
          wilson_lower_bound(470, 500) > wilson_lower_bound(47, 50),
          "%.4f vs %.4f" % (wilson_lower_bound(470, 500), wilson_lower_bound(47, 50)))


@case
def test_runs_needed_is_monotonic():
    a = runs_needed_for(0.90)
    b = runs_needed_for(0.95)
    c = runs_needed_for(0.99)
    check("harder bars need more runs", a < b < c, "%s %s %s" % (a, b, c))
    check("0.95 needs a real number of runs", 50 < b < 100, str(b))
    print("       consecutive passes needed for a 0.90/0.95/0.99 lower bound: %d/%d/%d"
          % (a, b, c))


@case
def test_broken_is_not_flaky():
    histories = {
        "solid": [True] * 30,
        "flappy": [True, False] * 15,
        "always_fails": [False] * 30,
        "barely_run": [True, False],
    }
    quarantined, classes = quarantine_set(histories)
    check("stable", classes["solid"] == "stable")
    check("flaky", classes["flappy"] == "flaky")
    check("a test that always fails is broken, not flaky",
          classes["always_fails"] == "broken", classes["always_fails"])
    check("not enough history", classes["barely_run"] == "unknown")
    check("only the flaky one is quarantined", quarantined == {"flappy"}, str(quarantined))


@case
def test_gate_blocks_the_things_it_should():
    histories = {"a": [True] * 30, "b": [True] * 30, "c": [False] * 30}
    results = {"a": True, "b": True, "c": False}
    verdict = qualify(results, histories)
    check("a consistently failing test blocks", not verdict["qualified"], str(verdict))
    check("and is named as broken", verdict["broken"] == ["c"], str(verdict["broken"]))

    small = qualify({"a": True}, {"a": [True] * 30}, required_lower_bound=0.95)
    check("one passing test is not a qualification", not small["qualified"], str(small))
    check("raw rate would have said yes", small["raw_pass_rate"] == 1.0)

    many_h = {"t%d" % i: [True] * 30 for i in range(200)}
    many_r = {"t%d" % i: True for i in range(200)}
    big = qualify(many_r, many_h, required_lower_bound=0.95)
    check("200 passing tests qualifies", big["qualified"], str(big["reasons"]))

    blocked = qualify(many_r, many_h, required_lower_bound=0.95, unschedulable=3)
    check("cases with no capable bench block the build", not blocked["qualified"],
          str(blocked["reasons"]))


@case
def test_quarantine_does_not_rescue_a_bad_build():
    histories = {"t%d" % i: [True] * 30 for i in range(50)}
    histories["flappy"] = [True, False] * 15
    results = {"t%d" % i: (i > 8) for i in range(50)}
    results["flappy"] = False
    verdict = qualify(results, histories, required_lower_bound=0.95)
    check("real failures still block even with a quarantine list",
          not verdict["qualified"], str(verdict))
    check("the flaky test was excluded", "flappy" in verdict["quarantined"])
    check("but the count only dropped by one", verdict["counted_tests"] == 50,
          str(verdict["counted_tests"]))


def main():
    print("\n%d cases passed, %d assertions, %d failures, %.2fs"
          % (R["passed"], R["assertions"], R["failed"], time.time() - _T0))
    if R["failures"]:
        for f in R["failures"]:
            print("  - %s" % f)
        sys.exit(1)


if __name__ == "__main__":
    main()
