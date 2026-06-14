"""
Hydra submit wrapper for dataset sweeps.

By default, multirun sweeps are launched once per dataset. Optionally,
`single_array_across_datasets=true` submits one multirun that sweeps
`dataset_index` across all configured datasets in a single Slurm array.
"""
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import subprocess
import sys
from pathlib import Path
import grok
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from job_sub.utils.config_utils import (
    # load_datasets_or_raise,
    parse_override_value,
    # seed_env_from_datasets,
)
from job_sub.utils.seed_jobs import run_seed_jobs
from job_sub.utils.slurm_utils import wait_for_slurm_jobs
from job_sub.utils.sweep_utils import collect_user_overrides
# from run_grok_with_hydra import run_one_grok_with_hydra

_SCRIPT_PATH = Path(__file__).resolve()
# _HYDRA_CHILD_ENV = "GENE_CIRCUIT_HYDRA_CHILD"
# _DATASET_INDEX_ENV = "AL_DATASET_INDEX"
# _DATASET_ENV = "AL_DATASET_NAME"
# _METADATA_ENV = "AL_METADATA_PATH"
# _EMBED_DIR_ENV = "AL_EMBEDDING_ROOT"
# _SUBSET_ENV = "AL_SUBSET_IDS_PATH"
_CONFIG_PATH = _SCRIPT_PATH.parent / "conf" / "config.yaml"
# DATASETS, _ = load_datasets_or_raise(sys.argv[1:], _CONFIG_PATH)

_THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",#openMP is a common library for parallelism in C/C++ and Fortran, and setting OMP_NUM_THREADS to 1 limits it to using a single thread.
    "MKL_NUM_THREADS": "1",#MKL(Intel Math Kernel Library) is a library of optimized math routines for science, engineering, and financial applications. Setting MKL_NUM_THREADS to 1 limits it to using a single thread.
}#avoid CPU oversubscription
# _SINGLE_ARRAY_KEY = "single_array_across_datasets" #each dataset a seprate Slurm array, or one array sweeping across datasets via dataset_index


def _ensure_thread_env() -> None:
    """Cap per-process thread pools to avoid CPU oversubscription."""
    for key, value in _THREAD_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)#os.environ.setdefault() is a method in Python that sets the value of an environment variable if it is not already set. In this case, it ensures that the specified environment variables for thread limits are set to "1" if they haven't been defined elsewhere, preventing CPU oversubscription when running multiple processes in parallel.
    os.environ.setdefault("HYDRA_FULL_ERROR", "1") #set HYDRA_FULL_ERROR to "1" to get full stack traces from Hydra in case of errors, which can be helpful for debugging.


# seed_env_from_datasets(
#     DATASETS,
#     hydra_child_env=_HYDRA_CHILD_ENV,
#     dataset_env=_DATASET_ENV,
#     metadata_env=_METADATA_ENV,
#     embedding_env=_EMBED_DIR_ENV,
#     subset_env=_SUBSET_ENV,
# )
# os.environ.setdefault(_DATASET_INDEX_ENV, "0")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_one_job(cfg):
    """Run a single experiment with the given configuration (Hydra entrypoint)."""
    _ensure_thread_env()
    print(OmegaConf.to_yaml(cfg.al_settings))
    if OmegaConf.select(cfg, "seeds_as_jobs", default=False):
        grok.training.train(cfg)
    else:
        run_seed_jobs(cfg)


# def main():
    
    # """Launch multirun sweeps using either per-dataset or single-array mode."""
    # _ensure_thread_env()
    # user_overrides = collect_user_overrides(sys.argv[1:])
    # # if _single_array_across_datasets_enabled(user_overrides):
    # #     _run_single_array_sweep(user_overrides)
    # #     return

    # for dataset_index, dataset in enumerate(DATASETS):
    #     print(f"\n{'=' * 80}")
    #     print(f"Processing dataset: {dataset.name}")
    #     print(f"{'=' * 80}\n")
    #     env = os.environ.copy()
    #     env[_HYDRA_CHILD_ENV] = "1"
    #     env[_DATASET_INDEX_ENV] = str(dataset_index)
    #     env[_DATASET_ENV] = dataset.name
    #     env[_METADATA_ENV] = dataset.metadata_path
    #     env[_EMBED_DIR_ENV] = dataset.embedding_dir
    #     if dataset.subset_ids_path:
    #         env[_SUBSET_ENV] = dataset.subset_ids_path
    #     elif _SUBSET_ENV in env:
    #         env.pop(_SUBSET_ENV, None)
    #     cmd = [
    #         sys.executable,
    #         str(_SCRIPT_PATH),
    #         "-m",
    #         *user_overrides,
    #     ]
    #     _run_dataset_sweep(cmd, env, dataset.name)
    #     wait_for_slurm_jobs(dataset.name)




# def _run_single_array_sweep(user_overrides: list[str]) -> None:
#     """Launch one multirun sweep spanning all datasets (single Slurm array)."""
#     sweep_overrides = list(user_overrides)
#     if parse_override_value(sweep_overrides, "dataset_index") is None:
#         dataset_indices = ",".join(str(i) for i in range(len(DATASETS)))
#         sweep_overrides.append(f"dataset_index={dataset_indices}")
#     env = os.environ.copy()
#     env[_HYDRA_CHILD_ENV] = "1"
#     env.pop(_DATASET_ENV, None)
#     env.pop(_METADATA_ENV, None)
#     env.pop(_EMBED_DIR_ENV, None)
#     env.pop(_SUBSET_ENV, None)
#     env.pop(_DATASET_INDEX_ENV, None)
#     cmd = [sys.executable, str(_SCRIPT_PATH), "-m", *sweep_overrides]
#     _run_dataset_sweep(cmd, env, "ALL_DATASETS")


def _single_array_across_datasets_enabled(user_overrides: list[str]) -> bool:
    """Resolve whether single-array dataset sweeping is enabled."""
    override = parse_override_value(user_overrides, _SINGLE_ARRAY_KEY)
    if override is not None:
        return _parse_boolish(override, default=False)
    if not _CONFIG_PATH.exists():
        return False
    cfg = OmegaConf.load(_CONFIG_PATH)
    return _parse_boolish(cfg.get(_SINGLE_ARRAY_KEY), default=False)


def _parse_boolish(value, default: bool) -> bool:
    """Parse common truthy/falsy string values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


# def _run_dataset_sweep(cmd: list[str], env: dict, dataset_name: str) -> None:
#     """
#     Launch a Hydra multirun for a single dataset, suppressing transient SubmitIt errors.
#     """
#     try:
#         subprocess.run(
#             cmd,
#             check=True,
#             env=env,
#             stderr=subprocess.PIPE,
#             text=True,
#         )
#     except subprocess.CalledProcessError as exc:
#         stderr = exc.stderr or ""
#         if "submitit.core.utils.UncompletedJobError" in stderr:
#             print(
#                 f"[WARN] SubmitIt reported unfinished jobs for dataset '{dataset_name}'. Continuing.",
#                 file=sys.stderr,
#             )
#             return
#         if stderr:
#             print(
#                 f"[ERROR] Hydra sweep failed for dataset '{dataset_name}'. stderr:\n{stderr}",
#                 file=sys.stderr,
#             )
#         raise


if __name__ == "__main__":
    # if os.environ.get(_HYDRA_CHILD_ENV) == "1":
    #     # Already inside Hydra child: run once
    #     run_one_job()
    # elif any(flag in sys.argv[1:] for flag in ("-m", "--multirun")):
    #     # User requested multirun: fan out over data paths via subprocess
    #     main()
    # else:
    #     # Single run, no wrapper looping and no sweeps
    #     run_one_job()
    run_one_job()
