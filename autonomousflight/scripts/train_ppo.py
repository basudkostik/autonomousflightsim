"""
Phase C — PPO Training Script
===============================
Train a PPO agent using Stable-Baselines3 to navigate under canopy
and find/confirm fire in the AirSim forest environment.

USAGE:
  python scripts/train_ppo.py
  python scripts/train_ppo.py --timesteps 100000 --fire-x 50 --fire-y 50

PREREQUISITES:
  1. UE4 running with AirSim + Play pressed
  2. Fire actor placed in the scene
  3. pip install stable-baselines3 torch gymnasium
"""

import os
import sys
import time
import argparse
import numpy as np

# Fix Windows console encoding for emoji output
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from ppo_environment import AirSimFireEnv


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════
DEFAULT_TIMESTEPS = 50000

# PPO Hyperparameters
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
}

# Policy network architecture
POLICY_KWARGS = {
    "net_arch": {
        "pi": [128, 64],
        "vf": [128, 64]
    }
}


class TensorboardLogCallback(BaseCallback):
    """Custom callback for additional TensorBoard logging."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.fire_confirmations = 0
        self.collisions = 0
        self.total_episodes = 0

    def _on_step(self) -> bool:
        # Log episode info when available
        infos = self.locals.get("infos", [])
        for info in infos:
            if "termination" in info:
                self.total_episodes += 1
                if info["termination"] == "fire_confirmed":
                    self.fire_confirmations += 1
                elif info["termination"] == "collision":
                    self.collisions += 1

                # Log metrics
                if self.total_episodes > 0:
                    success_rate = self.fire_confirmations / self.total_episodes
                    collision_rate = self.collisions / self.total_episodes
                    self.logger.record("custom/success_rate", success_rate)
                    self.logger.record("custom/collision_rate", collision_rate)
                    self.logger.record("custom/total_episodes", self.total_episodes)
                    self.logger.record("custom/fire_confirmations", self.fire_confirmations)

                if "distance_to_fire" in info:
                    self.logger.record("custom/final_distance", info["distance_to_fire"])
                if "steps_to_confirm" in info:
                    self.logger.record("custom/steps_to_confirm", info["steps_to_confirm"])

        return True


class PrintProgressCallback(BaseCallback):
    """Print training progress periodically."""

    def __init__(self, print_freq=1000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            # Gather stats from logger
            print(f"\n📊 Step {self.num_timesteps:,} | "
                  f"Time: {time.strftime('%H:%M:%S')}")
        return True


def make_env():
    """Create a wrapped environment."""
    def _init():
        env = AirSimFireEnv(render_mode="human")
        env = Monitor(env)
        return env
    return _init


def train(timesteps, resume_from=None):
    """
    Train the PPO agent.

    Args:
        timesteps: Total training timesteps
        resume_from: Path to existing model to resume training
    """
    # Setup directories
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("  🧠 PPO TRAINING — UNDER-CANOPY FIRE FINDING")
    print("=" * 60)
    print(f"  📈 Total timesteps: {timesteps:,}")
    print(f"  💾 Model directory: {model_dir}")
    print(f"  📊 Log directory: {log_dir}")
    if resume_from:
        print(f"  🔄 Resuming from: {resume_from}")
    print("-" * 60)

    # Create environment
    print("🌲 Creating environment...")
    env = DummyVecEnv([make_env()])

    # Create or load model
    if resume_from and os.path.exists(resume_from):
        print(f"🔄 Loading existing model from {resume_from}")
        model = PPO.load(resume_from, env=env, **PPO_CONFIG)
    else:
        print("🆕 Creating new PPO model...")
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            policy_kwargs=POLICY_KWARGS,
            tensorboard_log=log_dir,
            **PPO_CONFIG
        )

    # Print model info
    print(f"  Policy: {model.policy.__class__.__name__}")
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=model_dir,
        name_prefix="ppo_forest"
    )

    tb_callback = TensorboardLogCallback()
    progress_callback = PrintProgressCallback(print_freq=2000)

    callbacks = [checkpoint_callback, tb_callback, progress_callback]

    # Train!
    print(f"\n🚀 Starting training at {time.strftime('%H:%M:%S')}...")
    print("   (Use TensorBoard to monitor: tensorboard --logdir logs/)")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\n⛔ Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()

    # Save final model
    final_path = os.path.join(model_dir, "ppo_forest_final")
    model.save(final_path)
    print(f"\n💾 Final model saved: {final_path}.zip")

    # Print final stats
    print("\n" + "=" * 60)
    print("  📊 TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Total timesteps: {model.num_timesteps:,}")
    print(f"  Total episodes: {tb_callback.total_episodes}")
    print(f"  Fire confirmations: {tb_callback.fire_confirmations}")
    print(f"  Collisions: {tb_callback.collisions}")
    if tb_callback.total_episodes > 0:
        sr = tb_callback.fire_confirmations / tb_callback.total_episodes * 100
        print(f"  Success rate: {sr:.1f}%")
    print("=" * 60)

    env.close()
    return final_path


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for fire finding")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS,
                        help=f"Total training timesteps (default: {DEFAULT_TIMESTEPS})")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model.zip to resume training from")
    args = parser.parse_args()

    train(args.timesteps, args.resume)


if __name__ == "__main__":
    main()
