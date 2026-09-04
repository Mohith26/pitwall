# Pitwall

Test infrastructure for a fleet of hardware in the loop benches: declaring a
suite, packing it onto scarce and non interchangeable machines, collapsing the
resulting pile of failures into the handful of bugs behind them, finding which
commit broke a flaky test, and deciding whether the build ships.

The theme running through all of it is that the obvious version of each of
these is wrong in a way that does not look wrong. A pass rate hides how much
evidence it rests on. A quarantine list hides a real regression. Bisection
silently searches the wrong half. Clustering welds two different bugs into one
group and still reports a plausible cluster count. Each of those is measured
here rather than asserted.

## Layout

```
pitwall/spec.py           declare a suite, validate it, expand the matrix
pitwall/scheduler.py      list scheduling onto a heterogeneous bench pool
pitwall/triage.py         normalise, MinHash, LSH, cluster failure logs
pitwall/culprit.py        finding the bad commit when the oracle is noisy
pitwall/qualification.py  Wilson interval gating and quarantine policy
pitwall/harness.py        seeded synthetic fleet, suites, and failure logs
pitwall/experiments.py
pitwall/test_pitwall.py
```

Standard library only. `python -m pitwall.test_pitwall`,
`python -m pitwall.experiments <section>`.

## Declaring a suite

Benches are not interchangeable. One has the current compute revision and a
radar rack, another has last revision's compute and a camera rig, one is the
only machine with the brake actuator attached. So cases declare the
capabilities they need and the scheduler matches, rather than a test
discovering on the bench that the hardware it wanted is somewhere else.

Validation is strict because the failure it prevents is silent. A typo in a
capability name produces a case that no bench can host, and a suite that
quietly ran 88% of its cases and passed all of them will report green. On a
600 case suite, a fleet of simulation only machines cannot host **259 of the
600**, and the gate treats unschedulable cases as a hard block rather than
excluding them from the denominator.

## Scheduling

List scheduling on unrelated machines with eligibility constraints, which is
NP-hard, so the question is how far from optimal the heuristic actually lands.

Three orderings, on 600 expanded cases across 8 benches:

| policy | makespan | utilisation | priority inversions |
| --- | --- | --- | --- |
| greedy, priority order | 3214.6 | 0.676 | 80 |
| longest processing time first | 3207.5 | 0.678 | 498 |
| longest first within priority band | 3208.3 | 0.678 | 65 |

The analytic lower bound is 3206.9, so all three are within 0.25% of a bound
nothing can beat, and the makespan question is basically settled by the
capability structure rather than by the policy. What separates them is
priority: pure longest first produces 498 inversions to save 0.9 units of
makespan, which is a bad trade when the highest priority band is the set of
tests somebody is waiting on. The default keeps priority bands intact and
sorts longest first inside them.

Against brute force optimal on 60 small instances the heuristic averages
**1.086x optimal with a worst case of 1.455x**, comfortably inside the
2 - 1/m = 1.667 list scheduling factor, though that bound is for identical
machines and does not formally cover this case.

The more useful output is where the fleet is actually tight. Total seconds of
work divided by the number of benches eligible to run it:

```
simulation      3206.9      compute_rev_c   2317.6
camera          1435.9      radar            947.0
lidar            619.0      brake_actuator   595.6
compute_rev_b    411.1
```

Adding a ninth bench does nothing unless it has `compute_rev_c`. That is a
fleet purchasing decision falling out of a scheduling model, which is most of
why the model is worth having.

## Triage

A nightly run produces failures by the thousand and root causes by the dozen.
The expensive part is working out that these 340 failures are one bug wearing
340 different sets of pointer values, timestamps, and bench names.

Pipeline: normalise the varying junk out, shingle into token trigrams, MinHash
to a fixed size signature, LSH banding to avoid comparing everything to
everything, then union find over surviving pairs.

**Normalisation is the whole thing.** With it, pairwise F1 over 5 seeds of 200
logs is **0.877**. Without it, **0.003**. Raw logs from one root cause are
never byte identical, so every log becomes its own cluster and the pipeline
produces exactly as much value as not running it.

MinHash is checked against the identity it depends on: the probability two
signatures agree at a position equals the Jaccard similarity of the underlying
sets. Over 25 random set pairs at 256 permutations the estimate tracks true
Jaccard within **0.08**.

### The bug that does not look like a bug

Plain transitive closure over a similarity graph is one log away from disaster.
Two genuinely different bugs need only a single log that resembles both, and
union find welds them together permanently. At a 0.45 threshold on one seed
that produced **a single cluster of 133 logs containing four distinct root
causes**, while the total cluster count still looked entirely plausible.
Nothing about the output says it is broken.

The fix is to require a triangle: an edge only merges if some third log is also
similar to both endpoints. A single bridging log has nothing to stand on, and
real clusters are dense enough that triangles are everywhere.

```
                        biggest cluster    root causes in it   edges refused
plain closure           133 logs           4                   0 of 1828
triangle support         42 logs           2                   18 of 1828 (0.98%)
```

**Refusing 1% of edges is the entire difference.** It also makes the whole
thing far less sensitive to the threshold, which matters because the threshold
is the parameter nobody will ever retune:

| threshold | F1 plain | F1 with support | precision plain | precision with support |
| --- | --- | --- | --- | --- |
| 0.35 | 0.376 | 0.673 | 0.237 | 0.572 |
| 0.40 | 0.598 | 0.848 | 0.465 | 0.842 |
| 0.45 | 0.706 | **0.877** | 0.605 | 0.941 |
| 0.50 | 0.873 | 0.861 | 0.904 | 0.997 |
| 0.55 | 0.860 | 0.780 | 0.997 | 0.997 |
| 0.60 | 0.756 | 0.669 | 1.000 | 1.000 |

Precision and recall are not equally important here and F1 slightly misleads
by treating them as if they were. Splitting one bug across three clusters
wastes three engineers' mornings. Merging three bugs into one cluster means two
of them get closed as duplicates of the third and ship. For a safety critical
program the 0.50 row, at 0.997 precision, is arguably the better operating
point than the 0.45 row that maximises F1.

### An optimisation that does not pay yet

LSH banding examines **9,180 of 79,800 possible pairs at 400 logs, 11.5%**, and
saves almost no wall clock: 1.35 seconds against 1.45 seconds exhaustive.
Signature construction dominates, not pair comparison, so the asymptotic win is
real and the constant is not there yet at this size. Worth keeping because the
corpus this exists for is thousands of logs, not four hundred, but worth
labelling honestly rather than presenting an 8.7x pair reduction as an 8.7x
speedup.

## Finding the bad commit

Bisection assumes a deterministic oracle. Run the test at the midpoint, believe
it, halve the range. On a flaky test one unlucky pass on a genuinely broken
commit sends the search into the half that contains nothing, and nothing ever
recovers it.

128 commits, a healthy test passing 95% of the time, and a broken test that
still passes some of the time:

| pass rate when broken | naive | majority of 5 | SPRT | SPRT runs |
| --- | --- | --- | --- | --- |
| 0% | 0.815 | 1.000 | 0.995 | 10.9 |
| 10% | 0.570 | 0.955 | 0.980 | 20.4 |
| 30% | 0.300 | 0.490 | 0.965 | 30.3 |
| 50% | 0.115 | 0.130 | 0.945 | 58.2 |

Naive bisection always spends exactly 7 test runs and is a coin flip by the
time the broken test passes half the time. Majority of 5 spends 35 runs at
every node whether the evidence is obvious or marginal, and still collapses at
30%. The SPRT version accumulates a log likelihood ratio at one commit until
Wald's sequential test crosses a threshold, then halves the range like a normal
bisection. It spends **11 runs when the signal is clean and 58 when it is
awful**, which is the right shape, and holds above 94% accuracy throughout.

The naive row at 0% is worth a second look. A perfectly deterministic broken
test still only gets 81.5%, because the *healthy* commits flake 5% of the time
and one false failure on the good side is just as fatal.

I also built the version I thought would be cleverest first: keep a full
posterior over which commit is the culprit and probe at the posterior median
after every single run. At 64 commits it reaches **0.683 accuracy using 400
test runs**, against SPRT's **0.950 using 27.4**. Moving the probe after one
observation throws away the value of that observation, since one sample barely
moves the posterior and the next probe is usually the same place, just reached
more expensively. It is still in the repo because being able to point at the
worse idea and say why is more useful than pretending I went straight to the
right one.

One implementation note that is easy to miss: the log likelihood ratios have to
be computed from probabilities clamped away from 0 and 1. A deterministic test
gives `p_pass_bad = 0`, `log(0)` is not a number the test can carry, and more
importantly an infinite ratio means one observation is treated as absolute
proof with no way back.

## Release gating

Two things this refuses to do.

**It will not gate on a raw pass rate.** 47 of 50 is 94% and so is 4700 of
5000, and they are not the same evidence:

```
47 of 50      Wilson lower bound 0.838
470 of 500                       0.916
4700 of 5000                     0.933
```

The gate uses the lower bound, so a build that has barely been tested cannot
pass by being lucky. That also makes the cost of a strict bar explicit: a 0.95
lower bound needs **73 consecutive passes**, and a 0.99 bound needs **381**. If
the fleet cannot supply that many runs, the gate is decoration.

**It will not quarantine a test just because it fails a lot.** A test that
fails every single time is not flaky, it is a regression, and quarantining it
is exactly how a real bug ships. Quarantine requires the test to have shown
both outcomes. In the worked example, two flaky tests are quarantined and the
consistently failing one blocks the build, at a raw pass rate of 0.997 that
would have sailed through a percentage gate.

Unschedulable cases block outright, for the same reason.

## Verification

`python -m pitwall.test_pitwall`: **22 cases, 425 assertions, 0 failures, 3.0s.**

Checked against something external wherever one exists: brute force optimal
makespan for the scheduler, a published worked Wilson interval (9 of 10 giving
0.5958 to 0.9821), the Jaccard identity for MinHash, and ground truth labels
for both clustering and culprit finding. The validation tests assert on the
specific silent failures rather than just that validation returns something.

## What I left out

- Benches never fail. Real bench flakiness, where the machine rather than the
  test is the problem, is its own triage category and needs a bench health
  signal the scheduler can react to.
- Tests are independent. Real suites have setup that several cases share and
  ordering effects that make failures depend on what ran before.
- The culprit model assumes a single culprit commit and a monotone step from
  healthy to broken. Two bugs in one range, or one that gets fixed and
  reintroduced, break that assumption and the search has no way to notice.
- No cost model on bench time, so the scheduler optimises makespan rather than
  anything about what the tests are worth.
