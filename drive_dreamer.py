#!/usr/bin/env python3
"""
drive_dreamer.py — Run a trained DreamerV3 checkpoint on the physical RC car.

This script builds a minimal Donkeycar vehicle pipeline:
  PiCamera → DreamerPilot → PCA9685 drivetrain

A web controller on port 8887 lets you switch modes at runtime:
  user  — manual control via web UI (or joystick with --js)
  local — DreamerV3 autopilot

Usage (on Raspberry Pi):
    python drive_dreamer.py
    python drive_dreamer.py --checkpoint dreamer/checkpoints/ckpt_00100000.pt
    python drive_dreamer.py --js          # PS5 joystick override

Do NOT modify manage.py or any Donkeycar file. This script is the sole
entry point for DreamerV3 on-car inference.
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import donkeycar as dk
from donkeycar.parts.pipe import Pipe

from dreamer.config import DreamerConfig
from dreamer.donkey_adapter import DreamerPilot


# ─────────────────────────────────────────────────────────────────────────────
# Drive-mode selector: picks pilot vs user output based on user/mode
# ─────────────────────────────────────────────────────────────────────────────

class DriveMode:
    """
    Selects between user (manual) and pilot (DreamerV3) outputs.

    user/mode == 'local'       → DreamerV3 drives
    user/mode == 'user'        → human drives
    user/mode == 'local_angle' → DreamerV3 angle, human throttle
    """
    def run(self, mode, user_angle, user_throttle, pilot_angle, pilot_throttle):
        pilot_angle    = pilot_angle    or 0.0
        pilot_throttle = pilot_throttle or 0.0
        if mode == 'local':
            return pilot_angle, pilot_throttle
        elif mode == 'local_angle':
            return pilot_angle, user_throttle
        return user_angle, user_throttle


# ─────────────────────────────────────────────────────────────────────────────
# Main vehicle loop
# ─────────────────────────────────────────────────────────────────────────────

def drive(checkpoint: str, use_joystick: bool = False) -> None:
    cfg = dk.load_config()
    V = dk.vehicle.Vehicle()

    # ── Camera ────────────────────────────────────────────────────────────────
    if cfg.CAMERA_TYPE == "PICAM":
        from donkeycar.parts.camera import PiCamera
        cam = PiCamera(
            image_w=cfg.IMAGE_W,
            image_h=cfg.IMAGE_H,
            image_d=cfg.IMAGE_DEPTH,
            framerate=cfg.CAMERA_FRAMERATE,
            vflip=getattr(cfg, 'CAMERA_VFLIP', False),
            hflip=getattr(cfg, 'CAMERA_HFLIP', False),
        )
    else:
        # Fallback for testing without hardware (e.g. USB webcam or mock)
        from donkeycar.parts.camera import MockCamera
        print("[drive_dreamer] WARNING: CAMERA_TYPE is not PICAM — using MockCamera.")
        cam = MockCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)

    V.add(cam, outputs=['cam/image_array'], threaded=True)

    # ── Web controller (mode switch + manual override at http://<pi>:8887) ───
    from donkeycar.parts.controller import LocalWebController
    ctr = LocalWebController(port=cfg.WEB_CONTROL_PORT, mode=cfg.DRIVE_MODE)
    V.add(
        ctr,
        inputs=['cam/image_array'],
        outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
        threaded=True,
    )

    # ── Optional joystick controller (overrides web controller outputs) ───────
    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
        from donkeycar.parts.controller import get_js_controller
        js = get_js_controller(cfg)
        V.add(
            js,
            outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
            threaded=True,
        )

    # ── DreamerV3 pilot ───────────────────────────────────────────────────────
    # Always runs CPU on Pi; ~50-80 ms per step is fine at 20 Hz.
    dreamer_cfg = DreamerConfig(device='cpu')
    pilot = DreamerPilot(dreamer_cfg, checkpoint_path=checkpoint)
    V.add(
        pilot,
        inputs=['cam/image_array'],
        outputs=['pilot/angle', 'pilot/throttle'],
    )

    # ── Drive-mode selector ───────────────────────────────────────────────────
    V.add(
        DriveMode(),
        inputs=['user/mode', 'user/angle', 'user/throttle',
                'pilot/angle', 'pilot/throttle'],
        outputs=['angle', 'throttle'],
    )

    # ── Drivetrain: PCA9685 over I2C ──────────────────────────────────────────
    from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

    steering_controller = PCA9685(
        cfg.STEERING_CHANNEL,
        cfg.PCA9685_I2C_ADDR,
        busnum=cfg.PCA9685_I2C_BUSNUM,
    )
    steering = PWMSteering(
        controller=steering_controller,
        left_pulse=cfg.STEERING_LEFT_PWM,
        right_pulse=cfg.STEERING_RIGHT_PWM,
    )

    throttle_controller = PCA9685(
        cfg.THROTTLE_CHANNEL,
        cfg.PCA9685_I2C_ADDR,
        busnum=cfg.PCA9685_I2C_BUSNUM,
    )
    throttle = PWMThrottle(
        controller=throttle_controller,
        max_pulse=cfg.THROTTLE_FORWARD_PWM,
        zero_pulse=cfg.THROTTLE_STOPPED_PWM,
        min_pulse=cfg.THROTTLE_REVERSE_PWM,
    )

    V.add(steering,  inputs=['angle'],    threaded=True)
    V.add(throttle,  inputs=['throttle'], threaded=True)

    # ── Start the loop ────────────────────────────────────────────────────────
    print(f"[drive_dreamer] Starting at {cfg.DRIVE_LOOP_HZ} Hz — "
          f"web UI at http://localhost:{cfg.WEB_CONTROL_PORT}")
    print(f"[drive_dreamer] Mode: {cfg.DRIVE_MODE}  "
          f"(switch to 'local' in the web UI to engage autopilot)")
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run DreamerV3 on the RC car")
    parser.add_argument(
        '--checkpoint',
        default='dreamer/checkpoints/latest.pt',
        help='Path to .pt checkpoint file (default: dreamer/checkpoints/latest.pt)',
    )
    parser.add_argument(
        '--js', action='store_true',
        help='Use PS5 / joystick controller for manual override',
    )
    args = parser.parse_args()
    drive(checkpoint=args.checkpoint, use_joystick=args.js)
