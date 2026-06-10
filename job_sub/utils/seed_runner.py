"""Helpers for running per-seed experiments in parallel worker processes."""

from typing import Any

from omegaconf import OmegaConf

from job_sub.utils.config_utils import ensure_resolvers
from run_active_learning import run_one_experiment


def run_seed_experiment(raw_cfg: dict[str, Any]) -> None:
    """Recreate DictConfig and run experiment (used by multiprocessing workers)."""
    ensure_resolvers()
    cfg = OmegaConf.create(raw_cfg)
    OmegaConf.resolve(cfg)
    run_one_experiment(cfg)
