# FUDGE-Suite

Current reference for goals, scope, and plan. Detail lives in the pointer docs at the bottom.

FUDGE-Suite is a plug-and-play security benchmark for federated unlearning (FU) algorithms.

## Goals

**Primary:** Audit FU algorithms for removal fidelity and security
regressions. Three parts:
- Harness tool where input is any FU algorithm, output is a standardized readout, with one
  validated working example per algorithm type.
- Eval criteria: does tested FU algo actually remove malicious influence (removal fidelity, via
  backdoor canaries and gap-to-RFS), does it preserve utility, at what efficiency, and can its
  deletion operation be weaponized (robustness).
- The rigor: both axes of the threats X algorithms matrix are validated first, so every cell is
  interpretable.

**Method (not a separate goal):** Finding the hidden assumptions attacks rely on, and where
realistic FL violates them, is the validation method, not a second contribution. Reproduce each
threat in its native setup, run a positive control, then port to realistic FL (the ladder below).
This rigor follows Arp et al., "Dos and Don'ts of Machine Learning in Computer Security" (ML security,
matching this project, not the LLM pitfalls literature). Any config-dependence it surfaces is reported
as a benchmark output under the primary goal, not argued as a standalone thesis.

The paper is an FU-algorithm **security** benchmark (removal fidelity + weaponization robustness), NOT
a generic FU performance benchmark (utility/forgetting/efficiency), which already exists.

## Core idea

**The subject under test is always the unlearning algorithm, never the backdoor.** A backdoor is a
measurable marker unique to one or more clients, so its survival after unlearning reveals whether
the algorithm removed that client. RFS (retrain from scratch) is ideal but computationally expensive; a fast approximate unlearner may leave residue, and that gap is what's measured. See concepts.md.

**Detection is out of scope.** Every unlearning request names a client that is already identified.
How that client was flagged, whether by a detector, an audit, or a GDPR request, is someone else's
problem. This is why attacks that compete on stealth are not our concern: we do not detect, we
remove. Given a known client, we ask two things: did removing it actually work (eradication), and
can the removal itself be weaponized (exploitation).

## Two taxonomies (coverage/representativity)

**Threat categories (rows), by the adversary's relationship to the unlearning operation:**

| Mode | Category | What it does | Adversary | Examples |
|---|---|---|---|---|
| Eradication | 1. Persistence | influence that should be removed survives | the removed client | DBA, Neurotoxin |
| Exploitation | 2. Weaponization | the deletion request is the attack | the requester | BadFU, FedMUA |
| Exploitation | 3. Leakage | the unlearning update leaks the deleted data | honest-but-curious server | FUIA (empty) |
| Exploitation | 4. Obstruction | a third party blocks the operation | a non-requesting client | AUA (empty) |

Rows 1 and 2 are in scope. Rows 3 and 4 are named in the taxonomy but scoped out of the
experimental results, for different reasons:

- **Obstruction** is scoped out because it is not a distinct adversary relationship. In Liu et
  al.'s threat matrix (Threats, Attacks, and Defenses in Machine Unlearning, IEEE OJ-CS 2025),
  the attacker roles are data contributors (R1), requesting users (R2), and accessible users
  (R3). Obstruction maps to none of them; it appears only as a goal (G4, resource exhaustion).
  It also has no canonical attack with public code, and the candidates target collaborative-
  recovery unlearners, which our server-side unlearners are not.
- **Leakage** is a real category (Liu's R3, post-unlearning phase P3) with published attacks
  (FUIA, DRAUN), but it does not fit this harness. Our threat-model contract assumes a malicious
  client: it builds poisoned trainsets and crafts client updates. A leakage adversary is an
  honest-but-curious server reading update deltas, and it is scored by reconstruction fidelity
  rather than accuracy or ASR. Supporting it means a second adversary interface and a second
  scorer family. Neither FUIA nor DRAUN has public code either.

MIA is the verification metric (did it forget), not a category.

**FU algorithm types (columns):**

The field's surveys converge on one top-level split: exact vs approximate. We keep that split and
subdivide approximate by mechanism, because mechanism, not guarantee label, determines how an
algorithm responds to an attack.

| Guarantee | Mechanism | Defined by | Our instance | Status |
|---|---|---|---|---|
| Exact | sharded retraining | retrain the data away, sharded | own client-sharded sequential version (SISA adapted to centralized FedAvg); no usable public code exists, see research_todo.md | to build |
| Approximate | gradient | gradient ascent on the forget set | PGA | green |
| Approximate | second-order (certified) | Newton-step / influence functions | to pick (own Newton-step/influence implementation on LiSSA machinery from fedmua; certified decentralized FU exists, arXiv 2601.06436, but is decentralized and out of scope) | EMPTY |
| Approximate | calibration | replay cached history, recalibrate updates | FedEraser | not green (dba collapses, neurotoxin timed out) |

RFS (retrain from scratch) is not a roster entry. It is the oracle: every entry above is measured
by its gap to RFS. It runs once per threat scenario and is reused across all algorithm columns.

Survey dimensions that are not columns are fixed by scope: the FL setting is centralized and
synchronous, a central server orchestrates FedAvg round by round, and decentralized/peer-to-peer
and asynchronous FU are out of scope; the
unlearning objective is always client-level; certified unlearning is a guarantee inside the
second-order family, not its own type; verifiable unlearning is not a type at all, verification
is what this benchmark does (the audit and scorers).

This roster is a **case study of the architecture**: exactly one validated example per type is required,
to show the benchmark discriminates across FU-algorithm families. It is not an exhaustive survey of FU
algorithms. Empty types stay empty until they have a green example.

## Methodology: validate both axes, then cross them

Every result is a cell (algorithm A, threat T). A cell is only interpretable if both are validated.
- **Threat positive control** = the attack is real, reproduced in its native setup: the authors' own
  repo and config. This applies to every threat model, eradication and exploitation alike.
- **Algorithm positive control** = the algorithm demonstrably unlearns an easy case (approaches RFS).
- Then run the cross: validated threats x validated algorithms.

For every threat model, validation is a ladder:
1. Run the attack in the authors' own repo, native setup. Confirm it fires - this is the positive control.
2. Move the same attack to our realistic FU setup. If it still fires, that is the result.
3. If it fails in the realistic setup, go back to the assumptions the authors state made their code
   work, and identify which realistic FL conditions violate them. That gap is the secondary-goal finding.

## What we measure

RFS (retrain from scratch without the target) is the baseline. Every metric is read as a gap-to-RFS:
how far the fast unlearner sits from the perfect one.

- **Removal fidelity (adversarial):** backdoor ASR gap-to-RFS. Did the planted backdoor drop to RFS's
  near-zero level, or does malicious influence survive.
- **Forgetting completeness (benign):** MIA-advantage gap-to-RFS. Is the target client's ordinary data
  as unrecognizable as under RFS, or still detectable as a member. Also covers benign (GDPR) deletion,
  where there is no backdoor to check.
- **Utility:** retained clean accuracy vs RFS. An unlearner that forgets by wrecking the model fails here.
- **Efficiency:** cost vs RFS by type: communication rounds (primary), compute, storage overhead, wall time.
- **Robustness:** resistance to weaponization (the exploitation modes).
  
## Plan

1. Finalize both rosters: threat categories and algorithm types.
2. Validate both axes (positive controls): threats fire/implant, algorithms unlearn an easy case.
   Every threat gets a native positive control in the authors' own repo, eradication and
   exploitation alike (BadFU and FedMUA have public code; DBA and Neurotoxin repos still to confirm).
3. Lock the measurements and the plug-and-play architecture around them.
4. Run the cross: validated threats x validated algorithms.
5. Report: the fidelity/robustness matrix, plus config-dependence wherever an exploitation attack
   broke moving from native to realistic FL.
   In parallel (mentor): keep writing (paper_draft.md).
6. Refresh README.md to match this plan (four-attack roster, the two axes, current framing). Currently stale.


## Open decisions

- Which instance fills the certified/second-order column: build our own Newton-step unlearner on
  the LiSSA machinery from fedmua, or adopt SIFU (arXiv 2211.11656), which has public code and
  runs FedAvg on CIFAR-10 but works by checkpoint rollback plus calibrated noise rather than
  second-order math. See research_todo.md for the full comparison.
- Whether the fixed setup in ARCHITECTURE.md is the right one: 50 clients, Dirichlet alpha 0.25,
  50 rounds, 10 of 50 clients per round, ResNet-18 on CIFAR-10. The axes are settled and
  documented; what needs a second opinion is whether these values are a defensible stand-in for
  realistic federated learning.
- Whether to add standard CIFAR-10 augmentation (random crop, horizontal flip). Training without
  it overfits more, which inflates the membership signal the MIA metric depends on. Deciding this
  after MIA is implemented means redoing the numbers, so it should be settled first.

## Pointers

- concepts.md: mental models (canary, why backdoors, goals, success criteria)
- notes_for_fudge_paper.md: detailed history and archive (native regimes, deviations, candidate attacks)
- README.md: how to run