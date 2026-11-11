import dataclasses
import torch
import os
import shutil
import warnings
from typing import Optional

class ResumeConfigWarning(UserWarning):
    """Raised when resuming from a checkpoint with a config value that differs from the one it was saved with"""

# Resuming with different values for these would silently corrupt the run: the RMS state, LR schedule and step
# accounting in the checkpoint all assume them
RESUME_MUST_MATCH = {
    "config":      {"num_envs"},
    "trainConfig": {"total_steps", "rollout_steps", "normalise_reward", "normalise_observations"},
}
# Bookkeeping fields that are expected to differ between a run and its resume
RESUME_IGNORE = {
    "config":      {"device", "run_dir"},
    "trainConfig": {"verbose", "checkpoint_every"},
}

class CheckpointManager():
    def __init__(self, trainer):
        self.trainer = trainer

        # Make space for checkpointing
        run_dir = "." if self.trainer.run_dir is None else self.trainer.run_dir
        # Sanitise only the env id (e.g. "ale_py:Breakout-v5"); run_dir may be an absolute Windows path like "C:\..."
        env_dir = self.trainer.env_id.replace(':', '_')
        self.checkpoints_path = os.path.join(run_dir, "checkpoints", env_dir, f"seed_{self.trainer.seed}")
        os.makedirs(self.checkpoints_path, exist_ok=True)

    def try_resume_from(self, resume_from: Optional[str]) -> tuple[bool, int]:
        is_resuming = False
        start_update = 0

        if resume_from == "latest":
            latest = os.path.join(self.checkpoints_path, "latest.pt")
            resume_from = latest if os.path.exists(latest) else None
        if resume_from is not None:
            start_update = self._load_checkpoint(resume_from)
            is_resuming = True

        return is_resuming, start_update

    def checkpoint(self, name, update, rollout_steps):
        latest = os.path.join(self.checkpoints_path, "latest.pt")
        self._save_checkpoint(latest, update=update, rollout_steps=rollout_steps)
        shutil.copyfile(latest, os.path.join(self.checkpoints_path, name))

    def _save_checkpoint(self, path, update, rollout_steps):
        checkpoint = {
            "actor":        self.trainer.actor.state_dict(),
            "critic":       self.trainer.critic.state_dict(),
            "optimiser":    self.trainer.opt.state_dict(),
            "lr_scheduler": self.trainer.lr_scheduler.state_dict(),
            "obs_rms":      self.trainer.obs_rms.state_dict() if self.trainer.obs_rms else None,
            "return_rms":   self.trainer.return_rms.state_dict() if self.trainer.return_rms else None,
            "update":       update + 1,
            "global_step":  (update + 1) * rollout_steps * self.trainer.num_envs,
            "config":       self._config_snapshot()
        }
        # Save to temp path and replace to reduce the risk of OS restarts while saving large models
        torch.save(checkpoint, path + ".tmp")
        os.replace(path + ".tmp", path)

    def _load_checkpoint(self, path):
        c = torch.load(path, map_location=self.trainer.device, weights_only=False)
        self._check_resume_config(c["config"])
        self.trainer.actor.load_state_dict(c["actor"])
        self.trainer.critic.load_state_dict(c["critic"])
        self.trainer.opt.load_state_dict(c["optimiser"])
        self.trainer.lr_scheduler.load_state_dict(c["lr_scheduler"])
        if self.trainer.obs_rms is not None:
            self.trainer.obs_rms.load_state_dict(c["obs_rms"])
        if self.trainer.return_rms is not None:
            self.trainer.return_rms.load_state_dict(c["return_rms"])

        start_update = c["update"]
        global_step = c["global_step"]
        config = c["config"]

        return start_update

    def _config_snapshot(self):
        as_dict = lambda cfg: {k: v for k, v in dataclasses.asdict(cfg).items() if not callable(v)}
        return {
            "config": as_dict(self.trainer.config),
            "trainConfig": as_dict(self.trainer.trainConfig),
            "ppoConfig": as_dict(self.trainer.ppoConfig)
            }

    def _check_resume_config(self, saved):
        errors, changed = [], []
        for section, current in self._config_snapshot().items():
            previous = saved.get(section, {})
            for k in sorted(current.keys() | previous.keys()):
                if k in RESUME_IGNORE.get(section, ()):
                    continue
                old, new = previous.get(k, "<missing>"), current.get(k, "<missing>")
                if old != new:
                    msg = f"{section}.{k}: checkpoint={old!r}, current={new!r}"
                    (errors if k in RESUME_MUST_MATCH.get(section, ()) else changed).append(msg)
        if errors:
            raise ValueError("Config is incompatible with checkpoint:\n  " + "\n  ".join(errors))
        for msg in changed:
            warnings.warn(f"Resuming with a changed config value, {msg}", ResumeConfigWarning)
