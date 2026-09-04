"""Every number in the README comes from here.

    python -m pitwall.experiments scheduler
    python -m pitwall.experiments triage
    python -m pitwall.experiments culprit
    python -m pitwall.experiments qualification
    python -m pitwall.experiments combine
"""

import json
import os
import random
import statistics
import sys
import time
from collections import Counter

from . import triage
from .culprit import bayesian_bisect, evaluate, naive_bisect, repeated_bisect, sprt_bisect
from .harness import make_failure_corpus, make_suite
from .qualification import qualify, runs_needed_for, wilson_lower_bound
from .scheduler import lower_bound, optimal_makespan, priority_inversions, schedule, utilization
from .spec import ALL_CAPABILITIES, default_fleet, expand, unschedulable, validate_suite

OUT_DIR = "results"


def _save(name, payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "part_%s.json" % name), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


def scheduler(seed=201):
    benches = default_fleet()
    cases = expand(make_suite(600, seed=seed))
    policies = {}
    t0 = time.time()
    for policy in ("greedy", "lpt", "constrained"):
        assignments, skipped, makespan = schedule(cases, benches, policy=policy)
        policies[policy] = {
            "makespan": round(makespan, 2),
            "utilization": round(utilization(assignments, benches, makespan), 4),
            "skipped": len(skipped),
            "priority_inversions": priority_inversions(assignments),
        }
    lb = lower_bound(cases, benches)
    elapsed = time.time() - t0

    rng = random.Random(seed)
    ratios = []
    solved = 0
    for _ in range(60):
        small_benches = default_fleet()[:3]
        small = expand(make_suite(7, seed=rng.randrange(100000)))
        for c in small:
            c.requires = frozenset()
        opt = optimal_makespan(small, small_benches, node_budget=200000)
        if opt is None:
            continue
        solved += 1
        _, _, got = schedule(small, small_benches)
        ratios.append(got / opt)

    # what does the fleet actually bottleneck on
    caps = Counter()
    for c in cases:
        for cap in c.requires:
            caps[cap] += c.duration
    hosts = {cap: sum(1 for b in benches if cap in b.capabilities) for cap in caps}
    pressure = {cap: round(caps[cap] / hosts[cap], 1) for cap in caps if hosts[cap]}

    sim_only = [b for b in benches if b.capabilities == frozenset(["simulation"])]
    orphans = unschedulable(cases, sim_only)

    return {
        "seed": seed,
        "cases": len(cases),
        "benches": len(benches),
        "lower_bound": round(lb, 2),
        "policies": policies,
        "best_policy": min(policies, key=lambda p: policies[p]["makespan"]),
        "best_over_lower_bound": round(
            min(p["makespan"] for p in policies.values()) / lb, 4),
        "schedule_seconds_for_three_policies": round(elapsed, 3),
        "vs_brute_force_optimal": {
            "instances_solved": solved,
            "mean_ratio": round(statistics.mean(ratios), 4),
            "worst_ratio": round(max(ratios), 4),
            "graham_bound_for_3_machines": round(2 - 1 / 3, 4),
        },
        "seconds_of_work_per_eligible_bench": pressure,
        "cases_a_simulation_only_fleet_could_not_host": len(orphans),
    }


def triage_experiment(seed=203, n_logs=200, seeds=(13, 17, 23, 29, 31)):
    grid = []
    for support in (False, True):
        for thr in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70):
            f1s, ps, rs, cs = [], [], [], []
            for s in seeds:
                logs, truth = make_failure_corpus(n_logs, seed=s)
                labels, _ = triage.cluster(logs, threshold=thr, require_support=support)
                sc = triage.pairwise_scores(labels, truth)
                f1s.append(sc["f1"]); ps.append(sc["precision"])
                rs.append(sc["recall"]); cs.append(sc["clusters_predicted"])
            grid.append({
                "require_support": support,
                "threshold": thr,
                "f1": round(statistics.mean(f1s), 4),
                "precision": round(statistics.mean(ps), 4),
                "recall": round(statistics.mean(rs), 4),
                "mean_clusters": round(statistics.mean(cs), 1),
            })
    best = max(grid, key=lambda r: r["f1"])

    logs, truth = make_failure_corpus(n_logs, seed=13)
    blobs = {}
    for support in (False, True):
        labels, stats = triage.cluster(logs, require_support=support)
        counts = Counter(labels)
        biggest = counts.most_common(1)[0][0]
        causes = {truth[i] for i, l in enumerate(labels) if l == biggest}
        blobs["require_support_%s" % support] = {
            "biggest_cluster_size": counts[biggest],
            "root_causes_inside_it": len(causes),
            "edges_above_threshold": stats["edges_above_threshold"],
            "edges_refused": stats["edges_refused_for_lack_of_support"],
            "refused_share": round(
                stats["edges_refused_for_lack_of_support"]
                / float(max(1, stats["edges_above_threshold"])), 4),
        }

    ablation = {}
    for label, kwargs in (("normalised", {}), ("raw_logs", {"do_normalize": False})):
        f1s = []
        for s in seeds:
            lg, tr = make_failure_corpus(n_logs, seed=s)
            f1s.append(triage.pairwise_scores(triage.cluster(lg, **kwargs)[0], tr)["f1"])
        ablation[label] = round(statistics.mean(f1s), 4)

    logs2, truth2 = make_failure_corpus(400, seed=41)
    t0 = time.time()
    _, lsh_stats = triage.cluster(logs2, use_lsh=True)
    lsh_s = time.time() - t0
    t0 = time.time()
    _, exact_stats = triage.cluster(logs2, use_lsh=False)
    exact_s = time.time() - t0

    return {
        "seed": seed,
        "logs_per_seed": n_logs,
        "seeds": list(seeds),
        "grid": grid,
        "best": best,
        "bridging_blob": blobs,
        "normalisation_ablation_f1": ablation,
        "lsh_cost": {
            "logs": 400,
            "possible_pairs": lsh_stats["possible_pairs"],
            "pairs_compared_with_lsh": lsh_stats["pairs_compared"],
            "pairs_compared_exhaustive": exact_stats["pairs_compared"],
            "fraction_of_pairs_examined": round(
                lsh_stats["pairs_compared"] / float(lsh_stats["possible_pairs"]), 4),
            "seconds_lsh": round(lsh_s, 2),
            "seconds_exhaustive": round(exact_s, 2),
        },
        "lsh_threshold_for_32_bands_of_4": round(triage.lsh_threshold(32, 4), 4),
    }


def culprit(seed=205, n_commits=128, trials=200):
    rows = []
    for p_pass_bad in (0.0, 0.10, 0.30, 0.50):
        entry = {"pass_rate_when_broken": p_pass_bad, "strategies": {}}
        for strategy, kw in (
            (naive_bisect, {}),
            (repeated_bisect, {"repeats": 5}),
            (sprt_bisect, {}),
        ):
            r = evaluate(strategy, n_commits, 0.95, p_pass_bad, trials=trials,
                         seed=seed, **kw)
            entry["strategies"][r["strategy"]] = {
                "accuracy": round(r["accuracy"], 4),
                "within_one_commit": round(r["within_one_commit"], 4),
                "mean_test_runs": round(r["mean_test_runs"], 2),
                "median_distance": r["median_distance"],
                "p95_distance": r["p95_distance"],
            }
        rows.append(entry)
    bayes = evaluate(bayesian_bisect, 64, 0.95, 0.30, trials=60, seed=seed)
    sprt64 = evaluate(sprt_bisect, 64, 0.95, 0.30, trials=60, seed=seed)
    return {
        "seed": seed,
        "commits_in_range": n_commits,
        "pass_rate_when_healthy": 0.95,
        "trials_per_cell": trials,
        "rows": rows,
        "posterior_median_bayesian_at_64_commits": {
            "accuracy": round(bayes["accuracy"], 4),
            "mean_test_runs": round(bayes["mean_test_runs"], 2),
        },
        "sprt_at_64_commits": {
            "accuracy": round(sprt64["accuracy"], 4),
            "mean_test_runs": round(sprt64["mean_test_runs"], 2),
        },
    }


def qualification(seed=207):
    runs = {("%.2f" % bar): runs_needed_for(bar) for bar in (0.80, 0.90, 0.95, 0.99)}
    same_rate = {
        "47_of_50": round(wilson_lower_bound(47, 50), 4),
        "470_of_500": round(wilson_lower_bound(470, 500), 4),
        "4700_of_5000": round(wilson_lower_bound(4700, 5000), 4),
    }
    histories = {"t%d" % i: [True] * 30 for i in range(300)}
    histories["flappy_a"] = [True, False] * 15
    histories["flappy_b"] = [True] * 20 + [False] * 6
    histories["always_broken"] = [False] * 30
    results = {k: True for k in histories}
    results["flappy_a"] = False
    results["flappy_b"] = False
    results["always_broken"] = False
    with_broken = qualify(results, histories)
    healthy = dict(results)
    healthy.pop("always_broken")
    hist2 = dict(histories)
    hist2.pop("always_broken")
    without_broken = qualify(healthy, hist2)
    orphaned = qualify(healthy, hist2, unschedulable=4)
    return {
        "seed": seed,
        "consecutive_passes_needed_for_lower_bound": runs,
        "same_94_percent_rate_different_evidence": same_rate,
        "gate_with_a_consistently_failing_test": {
            "qualified": with_broken["qualified"],
            "reasons": with_broken["reasons"],
            "raw_pass_rate": round(with_broken["raw_pass_rate"], 4),
            "wilson_lower": round(with_broken["wilson_lower"], 4),
            "quarantined": with_broken["quarantined"],
        },
        "gate_after_the_broken_test_is_fixed": {
            "qualified": without_broken["qualified"],
            "wilson_lower": round(without_broken["wilson_lower"], 4),
            "quarantined": without_broken["quarantined"],
        },
        "gate_with_unschedulable_cases": {
            "qualified": orphaned["qualified"],
            "reasons": orphaned["reasons"],
        },
    }


SECTIONS = {
    "scheduler": scheduler,
    "triage": triage_experiment,
    "culprit": culprit,
    "qualification": qualification,
}


def combine():
    out = {}
    for name in SECTIONS:
        path = os.path.join(OUT_DIR, "part_%s.json" % name)
        out[name] = json.load(open(path)) if os.path.exists(path) else None
    with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("wrote %s/results.json" % OUT_DIR)


def main():
    if len(sys.argv) < 2:
        print("sections: %s, combine" % ", ".join(sorted(SECTIONS)))
        return
    name = sys.argv[1]
    if name == "combine":
        combine()
        return
    t0 = time.time()
    _save(name, SECTIONS[name]())
    print("(%.1fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
