"""Helpers for running per-seed experiments in parallel worker processes."""

from argparse import Namespace
from typing import Any

from omegaconf import OmegaConf

from job_sub.utils.config_utils import ensure_resolvers


def run_seed_experiment(raw_cfg: dict[str, Any]) -> None:
    """Recreate DictConfig and run experiment (used by multiprocessing workers)."""
    ensure_resolvers()
    cfg = OmegaConf.create(raw_cfg)
    OmegaConf.resolve(cfg)
    import grok

    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise TypeError("Expected seed config to resolve to a mapping")
    grok.training.train(Namespace(**data))
