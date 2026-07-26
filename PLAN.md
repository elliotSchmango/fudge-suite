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

**Secondary:** Find hidden assumptions that these unlearning backdoors need, and why realistic FL violates them. Do this by reproducing threat models in their native setup and running a positive control.

The paper is an FU-algorithm **security** benchmark (removal fidelity + weaponization robustness), NOT
a generic FU performance benchmark (utility/forgetting/efficiency), which already exists.

## Core idea

**The subject under test is always the unlearning algorithm, never the backdoor.** A backdoor is a
measurable marker unique to one or more clients, so its survival after unlearning reveals whether
the algorithm removed that client. RFS (retrain from scratch) is ideal but computationally expensive; a fast approximate unlearner may leave residue, and that gap is what's measured. See concepts.md.

## Two taxonomies (coverage/representativity)

**Threat categories (rows), by the adversary's relationship to the unlearning operation:**

| Mode | Category | What it does | Adversary | Examples |
|---|---|---|---|---|
| Eradication | 1. Persistence | influence that should be removed survives | the removed client | DBA, Neurotoxin |
| Exploitation | 2. Weaponization | the deletion request is the attack | the requester | BadFU, FedMUA |
| Exploitation | 3. Leakage | the unlearning update leaks the deleted data | honest-but-curious server | FUIA (empty) |
| Exploitation | 4. Obstruction | a third party blocks the operation | a non-requesting client | AUA (empty) |

Current roster covers 1 and 2. MIA is the verification metric (did it forget), not a category.

**FU algorithm types (columns):**

| Type | Defined by | Our instance | Status |
|---|---|---|---|
| Exact / sharded | retrain the data away, sharded | SISA (to add); RFS is the control/oracle | to build |
| Gradient-based | gradient ascent on the forget set | PGA | green |
| Hessian-based | influence functions / Fisher / Newton-step | none | EMPTY (LiSSA machinery from fedmua reusable) |
| Calibration | replay cached history, recalibrate updates | FedEraser | not green (dba collapses, neurotoxin timed out) |

This roster is a **case study of the architecture**: exactly one validated example per type is required,
to show the benchmark discriminates across FU-algorithm families. It is not an exhaustive survey of FU
algorithms. Empty types stay empty until they have a green example.

## Methodology: validate both axes, then cross them

Every result is a cell (algorithm A, threat T). A cell is only interpretable if both are validated.
- **Threat positive control** = the attack is real. Eradication: implants a strong sticky backdoor.
  Exploitation: fires in its native setup (clone the authors' repo; the load-bearing validation).
- **Algorithm positive control** = the algorithm demonstrably unlearns an easy case (approaches RFS).
- Then run the cross: validated threats x validated algorithms.

For exploitation attacks, validation is a ladder:
1. Run the attack in the authors' own repo, native setup. Confirm it fires. This is the positive control.
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
   Exploitation positive controls run in the authors' own repos (BadFU, FedMUA have public code).
3. Lock the measurements and the plug-and-play architecture around them.
4. Run the cross: validated threats x validated algorithms.
5. Report: the fidelity/robustness matrix, plus config-dependence wherever an exploitation attack
   broke moving from native to realistic FL.
   In parallel (mentor): keep writing (paper_draft.md).
6. Refresh README.md to match this plan (four-attack roster, the two axes, current framing). Currently stale.


## Open decisions

- Whether to fill the empty threat categories (leakage via FUIA, obstruction via AUA) or scope them out
  with a stated reason.
- What defines the realistic FL/FU setup used for evaluation: dataset(s), model, client count, data
  split, participation. This is subjective, so it needs one stated, fixed definition. Every cell and
  every native-to-realistic comparison is read against the same setup.

## Pointers

- concepts.md: mental models (canary, why backdoors, goals, success criteria)
- notes_for_fudge_paper.md: detailed history and archive (native regimes, deviations, candidate attacks)
- README.md: how to run