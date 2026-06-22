# RAMP + NumericPDDLGym

This repository now contains a runnable RAMP-style training loop with:

- `RAMP+Rainbow` via `sb3-contrib` QR-DQN
- `RAMP+HER` only on goal-conditioned environments with native HER replay
- online AML updates through `NumericActionModelLearner`
- planner-first control with planner-to-RL fallback
- paper metrics: moving success rate (MA-25), cumulative solution length, and episode-to-90% efficiency

## Source libraries

This workspace builds on the following main libraries and upstream components:

- `stable-baselines3` for deep RL training, including DQN, PPO, and HER replay support
- `sb3-contrib` for the Rainbow-style QR-DQN implementation used here
- `gymnasium` for environment interfaces and wrappers
- local `NumericPDDLGym` sources under `NumericPDDLGym/` for PDDL-based Minecraft-style tasks
- local `numeric-sam` sources under `numeric-sam/` for numeric action-model-learning and planning-related tooling
- `pddl-plus-parser` for parsing and exporting PDDL domain/problem files
- local `Metric-FF` sources under `external/Metric-FF/Metric-FF-v2.1` for classical planning
- local `VAL` binary support under `external/VAL-build/` for plan validation workflows

## Run

```powershell
python experiments/train_ramp_rainbow.py --algorithm rainbow --difficulty easy --episodes 200 --seed 0
python experiments/train_ramp_rainbow.py --algorithm her --difficulty medium --episodes 200 --seed 1
```

## Fallback policy

This workspace is configured to preserve the paper's intended **planner → RL fallback** while removing fallbacks for missing implementations:

- planner-to-RL fallback remains enabled during episode control
- no HER fallback wrapper
- no AML fallback through `NSAMWrapper`

If a required capability is missing, the code should fail fast and report the missing dependency explicitly.

## Recommended setup: WSL for full planner/validator stack

Upstream `numeric-sam` tooling is Linux-oriented in practice. The recommended path for strict execution is WSL.

### 1. Install WSL (elevated PowerShell)

```powershell
wsl.exe --install -d Ubuntu
```

Reboot if prompted, complete Ubuntu first-run setup, then return to this repository.

### 2. Bootstrap Metric-FF and VAL inside WSL

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_wsl_planning_env.ps1
```

### 3. Verify the strict planning stack

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify_strict_planning_stack.ps1
```

## Metric-FF setup (native Windows, optional)

Metric-FF sources are extracted under `external/Metric-FF/Metric-FF-v2.1`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_metric_ff.ps1 -CheckOnly
powershell -ExecutionPolicy Bypass -File scripts/build_metric_ff.ps1
```

If tools are missing, install a MinGW/MSYS2 toolchain that provides `make`, `gcc`, `bison`, and `flex`.
This may help for local `Metric-FF`, but it is **not** the recommended path for full strict `numeric-sam` execution.

## Environment variables

`numeric-sam` expects these variables:

- `METRIC_FF_DIRECTORY`
- `ENHSP_FILE_PATH`
- `CONVEX_HULL_ERROR_PATH`
- `VALIDATOR_DIRECTORY`

Use the helper script to set them:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/set_planning_env.ps1
powershell -ExecutionPolicy Bypass -File scripts/set_planning_env.ps1 -Persist
```

Use `.env.example` as a template if you prefer to manage values manually.

## Current limitations

- `aml/nsam_wrapper.py` is strict and currently raises until direct `numeric-sam` observation conversion is implemented.
- Full upstream `numeric-sam` planner/validator execution is best run from WSL/Linux.
- For full deep RL runs, install dependencies from `requirements.txt` and editable local packages.
