# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**dreamer-car** is a self-driving RC car project combining:
- **Donkeycar v5.2.dev6** — RC car framework (manage.py, config.py, data collection)
- **DreamerV3** — world-model RL agent trained offline on tub data (`dreamer/`)

Hardware target: Raspberry Pi + PiCamera V2 (160×120 RGB @ 20Hz) + PCA9685 I2C servo controller.
Simulation: `donkey-warehouse-v0` via DonkeyGym on macOS.

---

## Setup (Fresh Clone)

Two external source repos are required alongside this one. Clone them first:

```bash
git clone https://github.com/autorope/donkeycar ~/donkeycar
git clone https://github.com/tawnkramer/gym-donkeycar ~/gym-donkeycar
```

Then set up the conda environment:

```bash
# 1. Create environment
conda create -n dreamer-car python=3.11 -y

# 2. Install PyTorch — choose ONE based on your hardware:
#    macOS (Apple Silicon, MPS):
conda run -n dreamer-car pip install torch torchvision

#    Linux CUDA 11.8:
conda run -n dreamer-car pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

#    Linux CUDA 12.1:
conda run -n dreamer-car pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

#    CPU only:
conda run -n dreamer-car pip install torch torchvision

# 3. Install all other dependencies
conda run -n dreamer-car pip install -r requirements.txt

# 4. Install donkeycar and gym-donkeycar as editable installs
#    (--no-deps because their deps are already covered by requirements.txt)
conda run -n dreamer-car pip install -e ~/donkeycar --no-deps
conda run -n dreamer-car pip install -e ~/gym-donkeycar --no-deps
```

Verify the environment:

```bash
conda run -n dreamer-car python -c "
import torch, donkeycar, gym_donkeycar
dev = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print('torch:', torch.__version__, '| device:', dev)
print('donkeycar:', donkeycar.__version__)
print('gym_donkeycar:', gym_donkeycar.__version__)
"
```

Smoke test:

```bash
conda run -n dreamer-car python dreamer/train.py --steps 100
```

> **Note:** `data/` is gitignored. The smoke test uses synthetic data automatically when `data/` is empty. To train on real data, collect tubs via `python manage.py drive` with the simulator running.

---

## All Commands

All commands use the `dreamer-car` conda env:

```bash
# Collect data / drive in simulator
conda run -n dreamer-car python manage.py drive

# Drive with PS5 controller
conda run -n dreamer-car python manage.py drive --js

# Train DreamerV3 offline on collected tubs
conda run -n dreamer-car python dreamer/train.py --tubs data/ --steps 100000

# SSH-safe: run in tmux so job survives disconnect
tmux new-session -d -s dreamer
tmux send-keys -t dreamer 'python dreamer/train.py --tubs data/ --steps 100000 2>&1 | tee dreamer/logs/train_$(date +%Y%m%d_%H%M%S).log' Enter
# Reattach later: tmux attach -t dreamer
# Check running:  tmux ls
# Kill session:   tmux kill-session -t dreamer

# Smoke test (100 steps, synthetic data)
conda run -n dreamer-car python dreamer/train.py --steps 100

# Calibrate servo / ESC (hardware only)
conda run -n dreamer-car python calibrate.py

# Test a checkpoint in the simulator (Shell 1: open sim, Shell 2: run this)
conda run -n dreamer-car python drive_dreamer_sim.py
conda run -n dreamer-car python drive_dreamer_sim.py --checkpoint dreamer/checkpoints/ckpt_00100000.pt
conda run -n dreamer-car python drive_dreamer_sim.py --episodes 5 --max-steps 2000

# Run DreamerV3 autopilot on the physical car (Raspberry Pi, no conda)
python drive_dreamer.py
python drive_dreamer.py --checkpoint dreamer/checkpoints/ckpt_00100000.pt
python drive_dreamer.py --js   # with PS5 joystick
```

---

## Quick Workflow (3 Commands)

```bash
# 1. Get the checkpoint from the remote GPU machine
scp 4060:~/harsh/Dreamer-donkeycar/dreamer/checkpoints/ckpt_00013000.pt ~/dreamer-car/dreamer/checkpoints/

# 2. Start the simulator (separate window)
open /Users/harshwadhawe/sim/DonkeySimMac/donkey_sim.app

pkill -f donkey_sim
# Run sim in tmux so it survives terminal close:
tmux new-session -d -s sim
tmux send-keys -t sim '~/sim/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim 2>&1 | tee dreamer/logs/sim_$(date +%Y%m%d_%H%M%S).log' Enter
# Reattach: tmux attach -t sim  |  Kill: tmux kill-session -t sim

# Linux GPU server sim:
tmux new-session -d -s sim
tmux send-keys -t sim './DonkeySimLinux/donkey_sim.x86_64 --port 9091 2>&1 | tee dreamer/logs/sim_$(date +%Y%m%d_%H%M%S).log' Enter

# 3. Run the autopilot against it
conda run -n dreamer-car python drive_dreamer_sim.py  --checkpoint dreamer/checkpoints/ckpt_00013000.pt
```

---

## Simulator Workflow

Environment: **`donkey-warehouse-v0`** (DonkeyGym, macOS).

Active `myconfig.py` settings:
```python
DONKEY_GYM = True
DONKEY_SIM_PATH = "/Users/harshwadhawe/sim/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim"
DONKEY_GYM_ENV_NAME = "donkey-warehouse-v0"
CONTROLLER_TYPE = 'ps5'
USE_JOYSTICK_AS_DEFAULT = False
```

**Shell 1** — start the simulator:
```bash
open /Users/harshwadhawe/sim/DonkeySimMac/donkey_sim.app
```

**Shell 2** — drive and collect tub data:
```bash
conda run -n dreamer-car python manage.py drive
```

> Donkeycar can also auto-launch the sim because `DONKEY_SIM_PATH` points to the binary. In that case Shell 1 is not needed.

Tub data layout (flat, directly in `data/`):
```
data/
  manifest.json           ← NDJSON metadata (not standard JSON)
  catalog_0.catalog       ← NDJSON records: {cam/image_array, user/angle, user/throttle}
  catalog_1.catalog
  ...
  images/
    0_cam_image_array_.jpg
    1_cam_image_array_.jpg
    ...
```

---

## Device Compatibility

Device is auto-detected at startup. Override with `--device cpu|cuda|mps`.

| Hardware | Device | Notes |
|---|---|---|
| Apple Silicon (M1/M2/M3) | `mps` | `torch.quantile` moved to CPU internally — MPS limitation |
| NVIDIA GPU | `cuda` | Use matching CUDA index-url when installing torch |
| Any | `cpu` | Automatic fallback |

---

## Donkeycar Architecture

`manage.py` builds a part pipeline run at 20Hz. Parts communicate via named keys in vehicle memory:

1. **Camera** → `cam/image_array`
2. **User Controller** (web UI / joystick) → `user/angle`, `user/throttle`, `user/mode`
3. **Drive Mode** — selects user vs pilot outputs
4. **AI Pilot** (optional `--model`) → `pilot/angle`, `pilot/throttle`
5. **Drivetrain** — converts angle/throttle to PWM on PCA9685
6. **Tub Writer** — saves `cam/image_array` + controls to `data/`

Config: `config.py` is read-only defaults. All overrides go in `myconfig.py` (uncomment lines there).

---

## Testing a Checkpoint in the Simulator (Current Workflow)

Training runs on a remote GPU machine. Checkpoints are SCP'd to this Mac and tested in the DonkeyGym simulator before any physical car deployment.

### Step 1 — Pull the checkpoint from the remote machine

```bash
# Run on this Mac — copies a specific checkpoint locally
scp <gpu-host>:~/dreamer-car/dreamer/checkpoints/ckpt_00100000.pt \
    ~/dreamer-car/dreamer/checkpoints/ckpt_00100000.pt
```

> Checkpoints are **not** in git (large binary files). Always copy via `scp`.
> `latest.pt` on the remote is a symlink — copy the actual `.pt` file by step number.

### Step 2 — Sync code changes via git

```bash
# On this Mac — push any code changes made here
git add -p && git commit -m "your message"
git push

# On the GPU machine — pull before the next training run
git pull
```

### Step 3 — Run the autopilot in the simulator

**Shell 1** — start the simulator:
```bash
open /Users/harshwadhawe/sim/DonkeySimMac/donkey_sim.app
```

**Shell 2** — run DreamerV3 against it:
```bash
# Default: latest.pt, 3 episodes, 1000 steps each
conda run -n dreamer-car python drive_dreamer_sim.py

# Specific checkpoint
conda run -n dreamer-car python drive_dreamer_sim.py \
    --checkpoint dreamer/checkpoints/ckpt_00100000.pt

# More episodes / longer runs
conda run -n dreamer-car python drive_dreamer_sim.py --episodes 10 --max-steps 2000
```

The script prints per-episode step count and total reward. Use this to compare checkpoints.

---

## Deploying to the Physical Car (Future)

Once a checkpoint passes simulator testing, deploy to the Raspberry Pi.

### Pi one-time setup

```bash
# On the Pi — CPU-only PyTorch (ARM64)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e ~/donkeycar --no-deps
```

### Copy checkpoint to the Pi

```bash
scp dreamer/checkpoints/ckpt_00100000.pt \
    pi@<pi-ip>:~/dreamer-car/dreamer/checkpoints/
```

### Calibrate (first time only)

```bash
python calibrate.py   # verify PWM values match config.py
```

### Run the autopilot

```bash
# On the Pi
python drive_dreamer.py --checkpoint dreamer/checkpoints/ckpt_00100000.pt
python drive_dreamer.py --js    # with PS5 joystick override
```

Open **`http://<pi-ip>:8887`** — switch mode to `local` to engage autopilot.

| Mode | Behaviour |
|---|---|
| `user` | Manual (web UI or joystick) |
| `local` | DreamerV3 full autopilot |
| `local_angle` | DreamerV3 steers, human throttle |

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: dreamer` | Run from `~/dreamer-car/` root |
| PCA9685 I2C error | `i2cdetect -y 1`; check `PCA9685_I2C_ADDR` in myconfig.py |
| Car steers wrong direction | Swap `STEERING_LEFT_PWM`/`STEERING_RIGHT_PWM` in myconfig.py |
| `checkpoint not found` warning | `latest.pt` is a symlink — copy the actual `.pt` file |

---

## DreamerV3 Architecture (`dreamer/`)

Fully isolated from Donkeycar files. Never modify `manage.py`, `config.py`, or any existing Donkeycar file.

| File | Responsibility |
|---|---|
| `dreamer/config.py` | `DreamerConfig` dataclass + `parse_config()` CLI override via argparse |
| `dreamer/world_model.py` | RSSM (GRU + 32×32 discrete latents), CNN encoder/decoder, reward & continuation heads |
| `dreamer/actor_critic.py` | Squashed-Gaussian actor, two-hot symlog critic, λ-returns, percentile return normalisation |
| `dreamer/replay_buffer.py` | Sequence replay buffer; `sample_batch()` → `(T, B, *)` tensors |
| `dreamer/envs/donkey_env.py` | Reads tub NDJSON catalogs → `ReplayBuffer`; synthetic fallback if no tubs |
| `dreamer/donkey_adapter.py` | Donkeycar part: `cam/image_array` → `pilot/angle`, `pilot/throttle` |
| `drive_dreamer_sim.py` | Simulator test loop: runs DreamerPilot against DonkeyGym, prints reward per episode |
| `drive_dreamer.py` | Physical car entry point: Donkeycar vehicle with DreamerPilot + PCA9685 drivetrain (Pi) |
| `dreamer/logger.py` | WandB + TensorBoard logging, GIF video saving |
| `dreamer/train.py` | Entry point: loads tubs → trains WM → trains AC → saves checkpoints |

Checkpoints: `dreamer/checkpoints/ckpt_<step>.pt`, symlinked as `latest.pt`.

Key paper details implemented:
- Discrete latents: 32 categories × 32 classes, straight-through gradients
- KL balancing: 80% prior / 20% posterior, free-bits clipped at 1 nat
- Symlog on observations, rewards, value targets
- Two-hot critic over 255 symlog-spaced bins
- Return normalisation: EMA 5th–95th percentile
- Gradient clipping: norm 1000 (WM), norm 100 (AC)
- Imagination horizon H=15, λ=0.95
