#!/usr/bin/env python
# coding: utf-8
import csv
import json
import logging
import os
import subprocess
from argparse import ArgumentParser
from copy import deepcopy
from glob import glob
from pathlib import Path
from pprint import pprint

import blobfile as bf
import grok
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import torch
import yaml
from tqdm import tqdm
import textwrap

logger = logging.getLogger(__name__)

# take args: input_dir output_dir
parser = ArgumentParser()
parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    required=True,
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    required=True,
)
parser = grok.training.add_args(parser)
args = parser.parse_args()
print(args, flush=True)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def load_expt_metrics(
    logger_dir,
    args,
):
    """load the metrics for one experiment"""
    args = deepcopy(args)

    # load the hparams for this experiment
    with open(f"{logger_dir}/hparams.yaml", "r") as fh:
        hparams_dict = yaml.safe_load(fh)

    for k, v in hparams_dict.items():
        setattr(args, k, v)

    # load the summarized validation and training data for every epoch
    val_data = {
        "step": [],
        "epoch": [],
        "val_loss": [],
        "val_accuracy": [],
        "full_train_acc": [],
    }
    train_data = {
        "step": [],
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "learning_rate": [],
    }

    with open(f"{logger_dir}/metrics.csv", "r") as fh:
        for row in csv.DictReader(fh):
            if row["train_loss"] != "":
                for k in train_data:
                    if k in ["step", "epoch"]:
                        v = int(row[k])
                    else:
                        v = float(row[k])
                    train_data[k].append(v)
            else:
                for k in val_data:
                    if k not in row or row[k] == "":
                        continue
                    if k in ["step", "epoch"]:
                        v = int(row[k])
                    else:
                        v = float(row[k])
                    val_data[k].append(v)

    return {
        "hparams": hparams_dict,
        "train": train_data,
        "val": val_data,
        # "raw": raw_data,
    }


def load_run_metrics(
    run_dir,
    args=args,
):
    """load all the metrics for a collection of experiments with the same architecture
    across various amounts of training data"""
    metric_data = {}
    logger_dirs = find_logger_dirs(run_dir)
    for logger_dir in tqdm(logger_dirs, unit="expt"):
        try:
            expt_data = load_expt_metrics(logger_dir, args)
            label = make_expt_label(run_dir, logger_dir)
            metric_data[label] = expt_data #edited to let it use the label instead of only train_data_pct as the key
        except FileNotFoundError:
            pass
    return metric_data


def find_logger_dirs(run_dir):
    """Find Lightning logger directories under a Hydra output tree."""
    run_path = Path(run_dir)
    metrics_files = sorted(run_path.glob("**/metrics.csv"))
    logger_dirs = []
    for metrics_file in metrics_files:
        logger_dir = metrics_file.parent
        if (logger_dir / "hparams.yaml").exists():
            logger_dirs.append(str(logger_dir))
    return logger_dirs


def make_expt_label(run_dir, logger_dir):
    """Use hydra folder plus seed folder as the plot label."""
    run_path = Path(run_dir).resolve()
    logger_path = Path(logger_dir).resolve()
    try:
        rel_parts = logger_path.relative_to(run_path).parts
    except ValueError:
        rel_parts = logger_path.parts

    seed = next((part for part in rel_parts if part.startswith("seed_")), None)
    if seed is not None:
        seed_index = rel_parts.index(seed)
        hydra_name = rel_parts[seed_index - 1] if seed_index > 0 else "run"
        return f"{hydra_name}_{seed}"

    ignored = {"lightning_logs"}
    parts = [part for part in rel_parts if part not in ignored and not part.startswith("version_")]
    return "_".join(parts) or logger_path.name


def add_metric_subplot(
    ax,
    setting_label,
    setting_data,
    metric_group,
    metric_specs,
    scales,
    by="step",  # step or epoch
    max_increment=0,
):
    ax.set_xscale(scales["x"])
    ax.set_yscale(scales["y"])
    ax.set_xlabel(by)
    ax.set_title("\n".join(textwrap.wrap(setting_label, width=42)))

    if "accuracy" in metric_group:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ymin = 1e-16
        ymax = 101
        ax.axis(ymin=ymin, ymax=ymax)
    if "loss" in metric_group:
        ymin = 1e-16
        ymax = 15
        ax.axis(ymin=ymin, ymax=ymax)

    for split, metric, color, label_name in metric_specs:
        logger.debug(f"processing {metric}")
        this_data = setting_data[split]
        X = this_data[by]
        Y = this_data[metric]
        if max_increment > 0:
            X = [x for x in X if x <= max_increment]
            Y = Y[: len(X)]

        if len(X) != len(Y):
            logger.warning(
                f"Mismatched data: {metric} at setting {setting_label} has "
                f"{len(X)} {by}s but {len(Y)} metric values"
            )
            continue
        if not Y:
            logger.warning(f"No data for {metric} at setting {setting_label}")
            continue

        label = label_name
        if "accuracy" in metric:
            label += " (max = %.2f)" % max(Y)
        elif "loss" in metric:
            label += " (min = %.2f)" % min(Y)
        ax.plot(X, Y, label=label, color=color)
    ax.legend()


# def add_max_accuracy_graph(
#     ax,
#     arch,
#     metric,
#     metric_data,
#     scales,
#     by="step",
#     max_increment=0,
# ):
#     ax.set_title(f"max {metric}")
#     ax.set_xlabel("experiment")
#     ymin = 1e-16
#     ymax = 101
#     ax.axis(ymin=ymin, ymax=ymax)
#     ax.set_xscale(scales["x"])
#     ax.set_yscale(scales["y"])
#     ax.yaxis.set_major_formatter(mtick.PercentFormatter())

#     T = list(sorted(metric_data.keys()))
#     Y = []
#     for i, t in enumerate(T):
#         if "val" in metric:
#             this_data = metric_data[t]["val"]
#         else:
#             this_data = metric_data[t]["train"]
#         X = this_data[by]
#         if max_increment > 0:
#             X = [x for x in X if x <= max_increment]
#             max_idx = len(X)
#         else:
#             max_idx = -1
#         try:
#             Y.append(max(this_data[metric][:max_idx]))
#         except ValueError:
#             Y.append(np.nan)

#     X = np.arange(len(T))
#     ax.set_xticks(X)
#     ax.set_xticklabels(T, rotation=45, ha="right")
#     label = f"max {metric} {arch}"
#     ax.plot(X, Y, label=label)


def create_loss_curves(
    metric_data,
    arch,
    operation,
    # epochs,
    most_interesting_only=False,
    image_dir=args.output_dir,
    by="step",
    max_increment=0,
    cmap="viridis",
):
    scales = {
        "x": "log",
        "y": "linear",
    }
    metric_groups = {
        "loss": [
            ("val", "val_loss", "tab:blue", "val"),
            ("train", "train_loss", "tab:orange", "train"),
        ],
        "accuracy": [
            ("val", "val_accuracy", "tab:blue", "val"),
            ("val", "full_train_acc", "tab:orange", "full train"),
        ],
        "learning_rate": [
            ("train", "learning_rate", "tab:green", "train"),
        ],
    }
    settings = list(sorted(metric_data.items()))
    if not settings:
        logger.warning("No settings found for loss curve plotting")
        return

    ncols = min(3, len(settings))
    nrows = int(np.ceil(len(settings) / ncols))
    fig_width = ncols * 7
    fig_height = nrows * 4.5

    for metric_group, metric_specs in metric_groups.items():
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(fig_width, fig_height),
            squeeze=False,
        )
        flat_axes = axs.flatten()
        for ax, (setting_label, setting_data) in zip(flat_axes, settings):
            add_metric_subplot(
                ax,
                setting_label,
                setting_data,
                metric_group,
                metric_specs,
                scales,
                by=by,
                max_increment=max_increment,
            )
        for ax in flat_axes[len(settings) :]:
            ax.axis("off")

        fig.suptitle(f"{operation} {metric_group} {arch} {max_increment:06d} {by}s")
        fig.tight_layout()

        img_file = (
            f"{image_dir}/loss_curves/{operation}_{metric_group}_{arch}"
            f"__upto_{max_increment:010d}_{by}"
        )
        if most_interesting_only:
            img_file += "_most_interesting"
        img_file += ".png"
        d = os.path.split(img_file)[0]
        os.makedirs(d, exist_ok=True)
        print(f"Writing {img_file}")
        fig.savefig(img_file)
        plt.close(fig)


def create_max_accuracy_curves(
    metric_data,
    arch,
    operation,
    by="step",
    max_increment=0,
    image_dir=args.output_dir,
):
    scales = {
        "x": "linear",
        "y": "linear",
    }

    ncols = 1
    nrows = 2
    fig_width = ncols * 8
    fig_height = nrows * 5
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))

    add_max_accuracy_graph(
        axs[0],
        arch,
        "val_accuracy",
        metric_data,
        scales,
        by=by,
        max_increment=max_increment,
    )
    axs[0].legend()
    add_max_accuracy_graph(
        axs[1],
        arch,
        "train_accuracy",
        metric_data,
        scales,
        by=by,
        max_increment=max_increment,
    )
    axs[1].legend()
    fig.suptitle(f"{operation} {arch} {max_increment:06d} {by}s")
    fig.tight_layout()

    img_file = f"{image_dir}/max_accuracy/{operation}_max_accuracy_{arch}_upto_{max_increment:010d}_{by}.png"
    d = os.path.split(img_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {img_file}")
    fig.savefig(img_file)
    plt.close(fig)


def create_tsne_graphs(
    operation,
    expt,
    run_dir,
    image_dir=args.output_dir,
):

    saved_pt_dir = f"{run_dir}/activations"
    saved_pts = []

    loss_ts = []
    accuracy_ts = []
    epochs_ts = []
    print(f'glob = {saved_pt_dir + "/activations_*.pt"}')
    files = sorted(glob.glob(saved_pt_dir + "/activations_*.pt"))
    print(f"files = {files}")

    for file in files:
        print(f"Loading {file}")
        saved_pt = torch.load(file)
        saved_pts.append(saved_pt)
        loss_ts.append(saved_pt["val_loss"].mean(dim=-1))
        accuracy_ts.append(saved_pt["val_accuracy"])
        epochs_ts.append(saved_pt["epochs"].squeeze())

    loss_t = torch.cat(loss_ts, dim=0).T.detach()
    accuracy_t = torch.cat(accuracy_ts, dim=0).T.detach()
    epochs_t = torch.cat(epochs_ts, dim=0).detach()
    print(loss_t.shape)
    print(accuracy_t.shape)
    print(epochs_t.shape)
    ######
    a = 0
    num_eqs = len(loss_t)
    b = a + num_eqs

    print("Doing T-SNE..")
    loss_tsne = TSNE(n_components=2, init="pca").fit_transform(loss_t)
    print("...done T-SNE.")

    ncols = 1
    nrows = 1
    fig_width = ncols * 8
    fig_height = nrows * 5
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))

    axs.scatter(loss_tsne[:, 0], loss_tsne[:, 1])

    img_file = f"{image_dir}/tsne/{operation}_{expt}.png"
    d = os.path.split(img_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {img_file}")
    fig.savefig(img_file)
    plt.close(fig)


def get_arch(metric_data):
    k = list(metric_data.keys())[0]
    hparams = metric_data[k]["hparams"]
    arch = f'L-{hparams["n_layers"]}_H-{hparams["n_heads"]}_D-{hparams["d_model"]}_B-{hparams["batchsize"]}_S-{hparams["random_seed"]}_DR-{hparams["dropout"]}'
    return arch


def get_operation(metric_data):
    k = list(metric_data.keys())[0]
    hparams = metric_data[k]["hparams"]
    operator = hparams["math_operator"]
    operand_length = hparams["operand_length"]
    _, operation = grok.data.ArithmeticDataset.get_file_path(operator, operand_length)
    return operation


def get_max_epochs(metric_data):
    k = list(metric_data.keys())[0]
    hparams = metric_data[k]["hparams"]
    return hparams["max_epochs"]


rundir = args.input_dir

try:
    metric_data = load_run_metrics(rundir, args)
    if not metric_data:
        raise FileNotFoundError(
            f"No metrics.csv with hparams.yaml found under {rundir}. "
            "Expected paths like seed_*/lightning_logs/version_*/metrics.csv."
        )
    arch = get_arch(metric_data)
    operation = get_operation(metric_data)
    max_epochs = get_max_epochs(metric_data)

    for by in ["step", "epoch"]:
        create_loss_curves(metric_data, arch, operation, by=by)

    # by = "epoch"
    # last_i = -1
    # for i in sorted(list(set(2 ** (np.arange(167) / 10)))):
    #     if max_epochs is not None and i > max_epochs:
    #         break
    #     i = int(round(i))
    #     create_max_accuracy_curves(
    #         metric_data,
    #         arch,
    #         operation,
    #         by=by,
    #         max_increment=i,
    #     )

    # make a video
    # in_files = os.path.join(
    #     args.output_dir,
    #     "max_accuracy",
    #     f"{operation}_max_accuracy_{arch}_upto_%*.png",
    # )
    # out_file = os.path.join(args.output_dir, f"{operation}_{arch}_max_accuracy.mp4")
    # cmd = [
    #     "ffmpeg",
    #     "-y",
    #     "-r",
    #     "16",
    #     "-i",
    #     in_files,
    #     "-vcodec",
    #     "libx264",
    #     "-crf",
    #     "25",
    #     "-pix_fmt",
    #     "yuv420p",
    #     out_file,
    # ]
    # subprocess.check_call(cmd)

except BaseException as e:
    print(f"{rundir} failed: {e}")
