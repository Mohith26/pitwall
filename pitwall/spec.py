"""A declarative description of what to run, and on what.

Hardware in the loop benches are not interchangeable. One has the current
compute revision and a radar rack, another has last revision's compute and a
camera rig, a third is the only one with the brake actuator hardware attached.
A test that needs radar cannot run on the camera bench, and finding that out by
having the test fail on the bench is an expensive way to learn it.

So a suite is declared rather than scripted: every case states the
capabilities it requires, its priority, its expected duration, and the axes it
should be expanded over. The expansion is deterministic and the validation is
strict, because the failure mode this exists to prevent is a typo in a
capability name silently producing a suite that schedules onto nothing and
reports green because it ran zero tests.

The format is a small typed dict tree, the kind of thing you would express as a
protobuf message in a real system. What matters here is the validate-then-
expand shape, not the serialisation.
"""

import itertools
import re

NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class SpecError(Exception):
    pass


class Bench:
    def __init__(self, bench_id, capabilities, throughput=1.0):
        self.bench_id = bench_id
        self.capabilities = frozenset(capabilities)
        self.throughput = throughput

    def __repr__(self):
        return "Bench(%s, %s)" % (self.bench_id, sorted(self.capabilities))


class TestCase:
    def __init__(self, name, requires, priority, duration, axes=None, retries=0):
        self.name = name
        self.requires = frozenset(requires)
        self.priority = priority
        self.duration = duration
        self.axes = axes or {}
        self.retries = retries

    def __repr__(self):
        return "TestCase(%s, p%d, %gs)" % (self.name, self.priority, self.duration)


def validate_suite(suite, known_capabilities):
    """Reject a suite before it costs bench time.

    Every check here corresponds to something that otherwise fails silently or
    fails late: an unknown capability produces an unschedulable case, a
    duplicate name makes results ambiguous, a non positive duration makes the
    scheduler's makespan meaningless, and an empty axis quietly deletes every
    expansion of that case.
    """
    problems = []
    seen = set()
    if not suite.get("cases"):
        problems.append("suite has no cases")
    for i, case in enumerate(suite.get("cases", [])):
        where = case.get("name", "cases[%d]" % i)
        name = case.get("name")
        if not name or not NAME_RE.match(name):
            problems.append("%s: name must match %s" % (where, NAME_RE.pattern))
        if name in seen:
            problems.append("%s: duplicate case name" % where)
        seen.add(name)
        requires = case.get("requires", [])
        for cap in requires:
            if cap not in known_capabilities:
                problems.append("%s: unknown capability %r" % (where, cap))
        if not isinstance(case.get("priority"), int) or case["priority"] < 0:
            problems.append("%s: priority must be a non negative integer" % where)
        duration = case.get("duration")
        if not isinstance(duration, (int, float)) or duration <= 0:
            problems.append("%s: duration must be positive" % where)
        for axis, values in (case.get("axes") or {}).items():
            if not NAME_RE.match(axis):
                problems.append("%s: axis name %r is not a valid identifier" % (where, axis))
            if not values:
                problems.append("%s: axis %r is empty, which deletes every expansion"
                                % (where, axis))
        retries = case.get("retries", 0)
        if not isinstance(retries, int) or retries < 0 or retries > 5:
            problems.append("%s: retries must be an integer in 0..5" % where)
    return problems


def expand(suite):
    """Cartesian product over each case's axes, in a stable order.

    Sorted axis names and sorted values, so the same suite always produces the
    same case list in the same order. Anything downstream that samples,
    schedules, or seeds from position depends on that being true.
    """
    out = []
    for case in suite["cases"]:
        axes = case.get("axes") or {}
        if not axes:
            out.append(TestCase(case["name"], case.get("requires", []), case["priority"],
                                case["duration"], retries=case.get("retries", 0)))
            continue
        keys = sorted(axes)
        for combo in itertools.product(*[sorted(axes[k]) for k in keys]):
            suffix = "".join("_%s_%s" % (k, v) for k, v in zip(keys, combo))
            out.append(TestCase(
                case["name"] + suffix,
                case.get("requires", []),
                case["priority"],
                case["duration"],
                axes=dict(zip(keys, combo)),
                retries=case.get("retries", 0),
            ))
    return out


def unschedulable(cases, benches):
    """Cases no bench in the pool can run.

    Reported separately from failures. A suite where 12% of the cases never ran
    because nothing could host them is not a passing suite, and a pass rate
    computed over only the cases that ran will say it is.
    """
    return [c for c in cases if not any(c.requires <= b.capabilities for b in benches)]


def default_fleet():
    """A small heterogeneous pool of the shape these fleets actually have: a
    couple of fully equipped benches that everything queues for, and several
    partial ones that only take a subset."""
    return [
        Bench("hil-01", ["compute_rev_c", "camera", "radar", "lidar", "brake_actuator"]),
        Bench("hil-02", ["compute_rev_c", "camera", "radar", "lidar"]),
        Bench("hil-03", ["compute_rev_c", "camera", "radar"]),
        Bench("hil-04", ["compute_rev_b", "camera", "radar", "lidar"]),
        Bench("hil-05", ["compute_rev_b", "camera"]),
        Bench("sim-01", ["simulation"], throughput=1.0),
        Bench("sim-02", ["simulation"], throughput=1.0),
        Bench("sim-03", ["simulation"], throughput=1.0),
    ]


ALL_CAPABILITIES = frozenset(
    c for b in default_fleet() for c in b.capabilities
)
