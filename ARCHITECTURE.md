# Architecture

How FUDGE-Suite is put together: what is fixed, what plugs in, and what a plug-in must provide.
For goals, scope, and the roster, see PLAN.md. For how to run it, see README.md.

## Fixed vs pluggable

The benchmark holds the federated learning setup still and varies only the parts under test.
Everything in the fixed column is a deliberate choice, not a default.

**Fixed** (changing any of these changes what the numbers mean, so they are not per-run options)

| Axis | Value | Where |
|---|---|---|
| Dataset | CIFAR-10 | `src/config.py`, normalization constants |
| Model | ResNet-18, CIFAR-adapted: 3x3 stride-1 stem, max-pool removed | `src/models/model.py` |
| Aggregation | FedAvg, so the setting is centralized and synchronous | `src/strategies/strategy.py` |
| Clients | 50 | `src/config.py` NUM_CLIENTS |
| Rounds | 50 | `src/config.py` NUM_ROUNDS |
| Data split | Dirichlet, alpha 0.25 | `src/config.py` DIRICHLET_ALPHA |
| Participation | partial, 10 of 50 clients per round | `fraction_fit` 0.2, `min_fit_clients` 10 |

**Pluggable** (the axes an experiment varies)

| Option | Default | Notes |
|---|---|---|
| `threat_model` / `threat_model_args` | `badfu` | the attack under test |
| `unlearner` / `unlearner_args` | `pga` | the FU algorithm under test |
| `scorers` | `accuracy`, `asr` | what gets measured |
| attack parameters | see below | strength and shape of the attack |
| `unlearn_client_id(s)` | `"0"` | which client the deletion request names |
| `seeds` | `[0]` | more than one seed reruns the pipeline and reports mean and std |

## Full experimental setup

Values in `ExperimentConfig` (`src/config.py`):

| Setting | Value |
|---|---|
| local epochs per round | 8 |
| client learning rate | 0.03, cosine decay to 0 |
| cooldown learning rate | 0.01 (durability probe only) |
| batch size | 64 |
| target label | 0 |
| poison ratio | 0.5 |
| amplification factor | 4.0 |
| trigger patch size | 3 |

Values hardcoded outside the config, listed here because they are experimental choices and not
visible to anyone reading `ExperimentConfig`:

| Setting | Value | Where |
|---|---|---|
| optimizer | SGD, momentum 0.9 | `src/client.py` |
| weight decay | 1e-4 | `src/client.py` |
| loss | cross-entropy | `src/client.py` |
| data augmentation | none: ToTensor and Normalize only | `src/runner.py` |

The missing augmentation is worth stating plainly. Standard CIFAR-10 training uses random crop and
horizontal flip. Training without them overfits more, and a more overfit model carries a stronger
membership signal, which inflates the MIA-advantage number used for forgetting completeness. Any
MIA result has to be read with that in mind, or augmentation has to be added first.

## Two setups, and which runs use which

Results come from two different federated setups. Every reported number needs to say which one it
came from, otherwise the comparison is not meaningful.

| Setup | Clients | Alpha | Participation | Partition file | Used for |
|---|---|---|---|---|---|
| Main harness | 50 | 0.25 | 10 of 50 per round | `partitions.json` | the cross run, threat x algorithm |
| BadFU native translation | 5 | 0.5 | full | `partitions_5c_a05.json` | matching the BadFU paper's regime |
| FedMUA native translation | 10 | 0.5 | full | `partitions_10c_a05.json` | matching the FedMUA paper's regime |

Native positive controls themselves run in the authors' own repositories, not here. The translation
setups above are the step in between: the authors' regime, rebuilt inside this harness.

## Contracts

A plug-in is registered by name (`src/registry.py`) and must satisfy one of three interfaces.

**Threat model** (`src/threat_models/base.py`). The adversary is a client. Required:

- `build_malicious_trainset(dataset, client_id)`: the poisoned training data for the attacking client
- `get_forget_set(dataset, client_id)`: the data the unlearner is asked to remove

Optional hooks cover multi-client attacks (`malicious_client_ids`, `is_malicious`), update shaping
(`craft_malicious_update`), sample-level targeting (`set_target_data`, `target_samples`), held-out
subpopulations (`holdout_indices`), and the honest-deletion control (`build_honest_forget_set`).

This contract is why the leakage threat category is out of scope. Every method above assumes the
adversary is a client that owns data and sends updates. A leakage adversary is an honest-but-curious
server that reads update deltas, and it is scored by reconstruction fidelity rather than by accuracy
or attack success rate. Supporting it needs a second adversary interface and a second scorer family,
not another entry in this one.

**Unlearner** (`src/unlearning/base.py`). Required:

- `name`: string, for telemetry
- `unlearn(model, forget_loader, retain_loader, context)`: returns the post-unlearning weights

`UnlearnContext` carries the global weights, client count, the client being unlearned, the device,
an optional per-round `history_cache` (only calibration-style unlearners need it, gated by
`cache_history`), and a `cost` dict for efficiency counters.

**Scorer** (`src/audit/scorers.py`). A scorer is built from the config and optionally the threat
model, then evaluates a set of weights. Current scorers: `accuracy`, `asr`, `misclassification`.
Note that MIA is listed as a metric in PLAN.md but is not implemented here yet.

## Code layout

| Path | Role |
|---|---|
| `src/runner.py` | runs the pipeline: train, RFS control, unlearn, audit, report |
| `src/training.py` | drives the Flower FedAvg simulation |
| `src/strategies/strategy.py` | FedAvg subclass, forces saboteurs into rounds, caches history |
| `src/client.py` | client-side local training |
| `src/registry.py` | plug-in registry for threat models, unlearners, scorers |
| `src/threat_models/` | BaseThreatModel and the attacks |
| `src/unlearning/` | BaseUnlearner plus rfs, pga, federaser |
| `src/audit/` | benchmarker and scorers |
| `src/datasets/` | partitioning, Dirichlet split, partition files |
| `src/models/model.py` | the locked model |
| `src/config.py` | ExperimentConfig and the fixed constants |
| `src/benchmark.py` | roster rows and the serial roster run |
| `src/main.py` | entry point, run modes |

## Baseline

RFS (retrain from scratch without the target client) is the oracle, not an entry in the roster.
Every metric is read as a gap to RFS. It depends only on the data, attack, and seed, not on which
unlearner is being tested, so it is computed once per threat scenario and reused across algorithms.
