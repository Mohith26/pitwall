"""Synthetic bench fleet and synthetic failures, all seeded.

Everything downstream needs data with known ground truth: which root cause a
failure came from, which commit actually broke a test, whether a given failure
was flakiness or a real regression. None of that is knowable from real logs
without hand labelling thousands of them, so it is generated here, with the
answer key kept separate from the data the algorithms see.

The failure templates are shaped like the things that actually go wrong on a
bench: a timeout waiting on a sensor frame, a checksum mismatch on a replayed
log, a watchdog reset, a clock skew between the bench and the unit under test,
a queue overflow when playback outruns consumption. Each one is decorated with
per run noise (addresses, timestamps, bench names, thread ids, jittered
numbers) that carries no diagnostic information and exists purely to defeat
naive string matching.
"""

import random

ROOT_CAUSES = [
    {
        "name": "sensor_frame_timeout",
        "lines": [
            "{ts} W sensor_bridge: no camera frame for {ms}ms on channel {ch}",
            "{ts} E sensor_bridge: frame deadline exceeded, expected {hz}Hz",
            "{ts} E harness: test aborted after sensor stall at t={t}s",
            "  at SensorBridge::WaitForFrame(/build/{path}/sensor_bridge.cc:{line})",
            "  at PlaybackRunner::Step(/build/{path}/playback_runner.cc:{line})",
        ],
    },
    {
        "name": "replay_checksum_mismatch",
        "lines": [
            "{ts} E replay: segment checksum mismatch, expected {addr} got {addr}",
            "{ts} E replay: dropping segment {n} of run {run}",
            "{ts} F harness: replay integrity check failed",
            "  at ReplayReader::VerifySegment(/build/{path}/replay_reader.cc:{line})",
        ],
    },
    {
        "name": "watchdog_reset",
        "lines": [
            "{ts} E watchdog: compute unit unresponsive for {ms}ms",
            "{ts} E watchdog: issuing hard reset on {bench}",
            "{ts} W harness: unit under test rebooted mid case, results discarded",
            "  at Watchdog::OnMissedHeartbeat(/build/{path}/watchdog.cc:{line})",
            "  at ThreadPool::Worker(tid={tid})",
        ],
    },
    {
        "name": "bench_clock_skew",
        "lines": [
            "{ts} W timesync: bench clock offset {float}ms exceeds tolerance",
            "{ts} E timesync: rejecting message with timestamp in the future",
            "{ts} E harness: correlation window empty, {n} messages discarded",
            "  at TimeSync::Correlate(/build/{path}/time_sync.cc:{line})",
        ],
    },
    {
        "name": "playback_queue_overflow",
        "lines": [
            "{ts} W playback: queue depth {n} exceeds high water mark",
            "{ts} E playback: dropped {n} lidar packets, consumer behind by {ms}ms",
            "{ts} E harness: point cloud incomplete, aborting",
            "  at PlaybackQueue::Push(/build/{path}/playback_queue.cc:{line})",
            "  at LidarDecoder::Decode(/build/{path}/lidar_decoder.cc:{line})",
        ],
    },
    {
        "name": "actuator_command_rejected",
        "lines": [
            "{ts} E actuator: brake command {float} outside commanded range",
            "{ts} E actuator: controller returned status {n} on {bench}",
            "{ts} F harness: safety interlock tripped",
            "  at BrakeActuator::Apply(/build/{path}/brake_actuator.cc:{line})",
        ],
    },
]

_PATHS = ["av/perception", "av/planner", "av/hal", "test/harness", "av/sensing"]


def _fill(template, rng):
    return (
        template.replace("{ts}", "2027-0%d-%02dT%02d:%02d:%02d.%03dZ" % (
            rng.randint(1, 9), rng.randint(1, 28), rng.randint(0, 23),
            rng.randint(0, 59), rng.randint(0, 59), rng.randint(0, 999)))
        .replace("{ms}", str(rng.randint(50, 9000)))
        .replace("{ch}", str(rng.randint(0, 7)))
        .replace("{hz}", str(rng.choice([10, 20, 30, 60])))
        .replace("{t}", str(round(rng.uniform(0.5, 900.0), 2)))
        .replace("{n}", str(rng.randint(1, 99999)))
        .replace("{float}", str(round(rng.uniform(-40.0, 40.0), 3)))
        .replace("{addr}", "0x%08x" % rng.getrandbits(32))
        .replace("{bench}", rng.choice(["hil-01", "hil-02", "hil-03", "hil-04", "sim-02"]))
        .replace("{run}", "run_%d" % rng.randint(100000, 999999))
        .replace("{tid}", str(rng.randint(1000, 9999)))
        .replace("{path}", rng.choice(_PATHS))
        .replace("{line}", str(rng.randint(40, 900)))
    )


def make_failure_log(root_cause_index, rng, noise_lines=2, drop_probability=0.15):
    """One failure log for a given root cause.

    Lines are dropped and unrelated lines interleaved at random, because real
    logs from one bug are not identical: verbosity differs, a retry adds
    output, an unrelated warning lands in the middle. A clustering approach
    that only works on byte identical logs is not doing anything.
    """
    cause = ROOT_CAUSES[root_cause_index]
    lines = [_fill(t, rng) for t in cause["lines"] if rng.random() > drop_probability]
    if not lines:
        lines = [_fill(cause["lines"][0], rng)]
    for _ in range(rng.randint(0, noise_lines)):
        other = ROOT_CAUSES[rng.randrange(len(ROOT_CAUSES))]
        lines.insert(rng.randint(0, len(lines)), _fill(rng.choice(other["lines"]), rng))
    return "\n".join(lines)


def make_failure_corpus(n_logs, seed, n_causes=None, noise_lines=2):
    """Returns (logs, truth_labels)."""
    rng = random.Random(seed)
    n_causes = n_causes or len(ROOT_CAUSES)
    logs = []
    truth = []
    for _ in range(n_logs):
        idx = rng.randrange(n_causes)
        logs.append(make_failure_log(idx, rng, noise_lines=noise_lines))
        truth.append(idx)
    return logs, truth


def make_suite(n_cases, seed):
    """A suite with a realistic capability mix: most cases run anywhere in
    simulation, a minority need specific hardware, and a few need the one bench
    with the brake actuator."""
    rng = random.Random(seed)
    profiles = [
        (["simulation"], 0.55),
        (["compute_rev_c", "camera"], 0.15),
        (["compute_rev_c", "camera", "radar"], 0.12),
        (["compute_rev_c", "camera", "radar", "lidar"], 0.10),
        (["compute_rev_c", "brake_actuator"], 0.05),
        (["compute_rev_b", "camera"], 0.03),
    ]
    cases = []
    for i in range(n_cases):
        r = rng.random()
        acc = 0.0
        chosen = profiles[0][0]
        for caps, weight in profiles:
            acc += weight
            if r <= acc:
                chosen = caps
                break
        cases.append({
            "name": "case_%04d" % i,
            "requires": chosen,
            "priority": rng.choices([0, 1, 2], weights=[0.15, 0.55, 0.30])[0],
            "duration": round(rng.lognormvariate(3.0, 0.9), 2),
            "retries": rng.choice([0, 0, 0, 1]),
        })
    return {"cases": cases}
