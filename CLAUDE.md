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

# Smoke test (100 steps, synthetic data)
conda run -n dreamer-car python dreamer/train.py --steps 100

# Calibrate servo / ESC (hardware only)
conda run -n dreamer-car python calibrate.py
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
