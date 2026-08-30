# FUDGE-Suite Plan

## Goal

Audit FU algorithms for removal fidelity and security regressions. Three parts:

- A harness where the input is any FU algorithm and the output is a standardized readout, with at least one
  validated working example per algorithm family.
- Evaluation criteria: does the algorithm actually remove malicious influence, does it preserve
  utility, at what cost, and can its deletion operation be weaponized.
- Rigor: both rosters are validated before they are crossed, so every result is interpretable.

This is an FU-algorithm security benchmark (removal fidelity and weaponization robustness), not a
generic FU performance benchmark, which already exists.

## Core idea

The subject under test is always the unlearning algorithm, never the backdoor. A backdoor is a
measurable marker unique to a client, so whether it survives unlearning reveals whether the
algorithm removed that client. RFS (retrain from scratch) is ideal but expensive. A fast
approximate unlearner may leave residue, and that gap is what we measure.

Detection is out of scope. Every unlearning request names a client that is already identified. How
it was identified is someone else's problem. Given a known client we ask two things: did removing it
actually work, and can the removal itself be weaponized.

## Rosters (finalized)

**Threats**

| Category | Adversary | Representative | Status |
|---|---|---|---|
| Persistence | the removed client | DBA | implemented, author code available |
| Weaponization | the requester | BadFU | implemented, currently parked, needs to manifest |
| Leakage | honest-but-curious server | none | scoped out |
| Obstruction | a non-requesting client | none | scoped out |

Neurotoxin and FedMUA stay implemented as secondary attacks in their categories. MIA is the
verification metric, not a threat category.

**Unlearning algorithms**

| Guarantee | Mechanism | Instance | Status |
|---|---|---|---|
| Exact | sharded retraining | own SISA-style build | to build |
| Approximate | gradient | PGA | green |
| Approximate | second-order (certified) | none | scoped out |
| Approximate | calibration | FedEraser | implemented, not green |

RFS is the oracle, not a roster entry. Every metric is read as a gap to RFS. It is computed once per
threat scenario and reused across algorithms.

Scoping decisions and their reasons are in design_reasoning.md. In short: leakage needs a
server-side adversary the threat contract does not support, obstruction is not a distinct adversary
relationship, and certified unlearning assumes convexity that ResNet-18 does not satisfy.

## What we measure

Every metric is read as a gap to RFS.

- **Removal fidelity:** backdoor ASR gap to RFS. Did the planted backdoor drop to the near-zero
  level RFS reaches, or does malicious influence survive.
- **Forgetting completeness:** MIA-advantage gap to RFS. Is the target client's ordinary data as
  unrecognizable as it is under RFS. Also covers benign deletion, where there is no backdoor.
- **Utility:** retained clean accuracy against RFS. An unlearner that forgets by wrecking the model
  fails here.
- **Efficiency:** cost against RFS. Communication rounds first, then compute, storage, wall time.
- **Robustness:** damage from an adversarial deletion compared to an honest deletion of the same
  shape. For BadFU this is the pre-unlearn to post-unlearn ASR swing. The baseline here is honest
  deletion, not RFS.

## Methodology

Every result is a cell (algorithm, threat). A cell is only interpretable if both are validated
first.

- **Threat positive control:** the attack is real, reproduced in its native setup using the authors'
  own repository and configuration. This applies to every threat.
- **Algorithm positive control:** the algorithm demonstrably unlearns an easy case, approaching RFS.
- Then run the cross.

For every threat, validation is a ladder:

1. Run the attack in the authors' own repository, native setup. Confirm it fires. This is the
   positive control.
2. Move the same attack to our setup. If it still fires, that is the result.
3. If it fails, identify which condition in our setup broke it, and report that as a finding.

Native reproduction is a one-time gate. It separates "the attack is real" from "our harness
reproduces it." Once a threat fires natively we work in our own setup and stop walking the authors'
code.

## Plan

1. **Rosters.** Done. Two threat categories, three algorithm families, six cells.
2. **Harness.** Settle data augmentation before anything else, since it moves every number.
   Implement MIA, which is a stated metric with no code behind it. Build the exact/sharded instance.
   Get FedEraser green.
3. **Native positive controls.** DBA and BadFU in the authors' repositories. Document their native
   regimes and build matching partition files.
4. **Algorithm positive controls.** Each unlearner unlearns an easy case and approaches RFS.
5. **Cross run.** Validated threats against validated algorithms, at least three seeds, mean and
   standard deviation.
6. **Report.** The fidelity and robustness matrix, plus config-dependence wherever an attack broke
   moving from its native setup to ours. Refresh README.md, which is stale.

## Open decisions

- Whether to add standard CIFAR-10 augmentation (random crop, horizontal flip). Training without it
  overfits more, which inflates the membership signal the MIA metric depends on. This has to be
  settled before MIA results mean anything.
- Whether the fixed setup in ARCHITECTURE.md is a defensible stand-in for realistic federated
  learning: 50 clients, Dirichlet alpha 0.25, 50 rounds, 10 of 50 clients per round, ResNet-18 on
  CIFAR-10.

## Known risks

- **FedEraser is the largest risk.** With second-order scoped out, three families remain and one of
  them does not work (DBA collapses, Neurotoxin times out). If it cannot be made green the roster
  falls to two working families, which is thin for a claim about discriminating across families.
- **BadFU is parked.** Its dormant-then-revived signature did not manifest, because influence was
  computed against a clean model that never learned the backdoor. Later commits may have moved this;
  it needs a run to confirm. See design_reasoning.md section 4.
- **Compute.** The cross run is federated ResNet-18 training with an RFS baseline per scenario and
  multiple seeds. It needs the cluster, not a laptop.

## Pointers

- ARCHITECTURE.md: what is fixed, what plugs in, the full experimental setup, plug-in contracts
- design_reasoning.md: why every scoping and roster decision was made, with citations (untracked)
- concepts.md: mental models (canary, why backdoors, success criteria)
- README.md: how to run
- archive/: superseded working documents
