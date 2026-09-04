"""Packing a suite onto a scarce, heterogeneous bench pool.

This is list scheduling on unrelated machines with eligibility constraints,
which is NP-hard, so the job is to pick a good heuristic and then find out how
far from optimal it actually lands rather than assuming.

Three policies are implemented and compared:

  greedy       cases in priority order, each to the capable bench that frees
               up soonest
  lpt          longest processing time first, the classic makespan heuristic,
               which ignores priority entirely
  constrained  longest first *within* each priority band, so urgent work still
               goes first but the long poles inside a band go early

Graham's bound says list scheduling on identical machines lands within
(2 - 1/m) of optimal. These machines are not identical, they have eligibility
constraints, so the bound does not formally apply. `experiments.py` measures
the real ratio against brute force optimal on instances small enough to solve
exactly.
"""



class Assignment:
    __slots__ = ("case", "bench_id", "start", "end")

    def __init__(self, case, bench_id, start, end):
        self.case = case
        self.bench_id = bench_id
        self.start = start
        self.end = end

    def __repr__(self):
        return "%s on %s [%.1f, %.1f)" % (self.case.name, self.bench_id, self.start, self.end)


def capable(bench, case):
    return case.requires <= bench.capabilities


def _order(cases, policy):
    if policy == "greedy":
        return sorted(cases, key=lambda c: (c.priority, c.name))
    if policy == "lpt":
        return sorted(cases, key=lambda c: (-c.duration, c.name))
    if policy == "constrained":
        return sorted(cases, key=lambda c: (c.priority, -c.duration, c.name))
    raise ValueError("unknown policy %r" % policy)


def schedule(cases, benches, policy="constrained"):
    """List scheduling. Returns assignments, skipped cases, and the makespan.

    Each case goes to whichever capable bench frees up soonest, which is a
    linear scan over the pool. That is O(cases * benches), and it is fine here
    because bench pools are tens of machines, not thousands. If the pool ever
    grew, the fix is a per capability heap keyed on free time rather than a
    cleverer policy.
    """
    free_at = {b.bench_id: 0.0 for b in benches}
    assignments = []
    skipped = []
    for case in _order(cases, policy):
        options = [b for b in benches if capable(b, case)]
        if not options:
            skipped.append(case)
            continue
        chosen = min(options, key=lambda b: (free_at[b.bench_id], b.bench_id))
        start = free_at[chosen.bench_id]
        end = start + case.duration / chosen.throughput
        free_at[chosen.bench_id] = end
        assignments.append(Assignment(case, chosen.bench_id, start, end))
    makespan = max((a.end for a in assignments), default=0.0)
    return assignments, skipped, makespan


def optimal_makespan(cases, benches, node_budget=2_000_000):
    """Exhaustive branch and bound over eligible assignments.

    Only tractable for small instances, which is exactly what it is for: a
    ground truth to measure the heuristics against. Returns None if the budget
    runs out, so those instances can be dropped rather than scored.
    """
    options = []
    for case in cases:
        capable_ids = [b.bench_id for b in benches if capable(b, case)]
        if not capable_ids:
            return None
        options.append((case, capable_ids))
    # longest first makes the bound bite early
    options.sort(key=lambda t: -t[0].duration)
    throughput = {b.bench_id: b.throughput for b in benches}
    best = [float("inf")]
    nodes = [0]

    def recurse(i, load):
        nodes[0] += 1
        if nodes[0] > node_budget:
            raise TimeoutError
        current = max(load.values()) if load else 0.0
        if current >= best[0]:
            return
        if i == len(options):
            best[0] = current
            return
        case, ids = options[i]
        seen_loads = set()
        for bench_id in ids:
            # symmetry break: two benches with the same current load and the
            # same eligibility are interchangeable, so only try one of them
            key = (load.get(bench_id, 0.0), bench_id in ids)
            if key in seen_loads:
                continue
            seen_loads.add(key)
            new_load = dict(load)
            new_load[bench_id] = new_load.get(bench_id, 0.0) + case.duration / throughput[bench_id]
            recurse(i + 1, new_load)

    try:
        recurse(0, {b.bench_id: 0.0 for b in benches})
    except TimeoutError:
        return None
    return best[0] if best[0] != float("inf") else None


def lower_bound(cases, benches):
    """Two bounds that hold for any schedule, whichever is larger.

    The longest single case, since nothing can finish before it does, and the
    total work divided across the benches eligible for it. The second bound is
    the one that catches an over-subscribed capability: if only one bench has
    the brake actuator, every brake test is serialised on it no matter how big
    the fleet is.
    """
    if not cases:
        return 0.0
    longest = max(c.duration for c in cases)
    by_capability = 0.0
    caps = set()
    for c in cases:
        caps |= set(c.requires)
    for cap in caps:
        work = sum(c.duration for c in cases if cap in c.requires)
        hosts = sum(1 for b in benches if cap in b.capabilities)
        if hosts:
            by_capability = max(by_capability, work / hosts)
    total = sum(c.duration for c in cases) / max(1, len(benches))
    return max(longest, total, by_capability)


def utilization(assignments, benches, makespan):
    if makespan <= 0:
        return 0.0
    busy = sum(a.end - a.start for a in assignments)
    return busy / (makespan * len(benches))


def priority_inversions(assignments):
    """How often a lower priority case started before a higher priority one.

    Some inversion is unavoidable and fine: a low priority case that only fits
    on an idle bench should take it rather than leave the bench empty. This
    counts them so the scheduler's behaviour is visible instead of assumed.
    """
    ordered = sorted(assignments, key=lambda a: (a.start, a.bench_id))
    count = 0
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if b.case.priority < a.case.priority and b.start > a.start:
                count += 1
                break
    return count
