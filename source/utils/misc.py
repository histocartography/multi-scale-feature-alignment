import importlib
import os
from typing import Dict
import numpy as np
import torch
import random

import collections
import gc
import resource
from source.da_models.abstract_da import AbstractDAModel

from source.utils.constants import LINE_LENGTH, MODELS_DIR

cudnn_deterministic = True


def debug_memory():
    print("maxrss = {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter(
        (str(o.device), o.dtype, tuple(o.shape))
        for o in gc.get_objects()
        if torch.is_tensor(o)
    )
    for line in tensors.items():
        print("{}\t{}".format(*line))


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False


def print_summary(scores: Dict[str, np.ndarray], weights: np.ndarray) -> None:
    """Print summary of results"""
    for name, metrics in zip(["Metric"], [scores]):
        print("\n")
        print(name)
        for name_metric, metric in metrics.items():
            print("*" * 108)
            print(name_metric)
            for i in range(metric.shape[0]):
                print("\t", end="")
                for j in range(metric.shape[1]):
                    print(f"{metric[i, j]:.4f}% ", end="")
                print(
                    f"\tAvg.:{np.average(metric[i, :], weights=weights):.4f}% ", end=""
                )
                print()
    print("*" * 108)


def read_args(argv, parser):
    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args(argv)
    args.results_path = os.path.expanduser(args.results_path)

    # Fix all seeds (cudnn, torch, numpy)
    seed_everything(seed=args.seed)

    print("=" * LINE_LENGTH)
    print("Arguments =")
    for arg in np.sort(list(vars(args).keys())):
        print("\t" + arg + ":", getattr(args, arg))
    print("=" * LINE_LENGTH)

    # Incremental Learning model
    if "generative" in args.model:
        DAModel = getattr(
            importlib.import_module(
                name=os.path.join(MODELS_DIR, args.model).replace("/", ".")
            ),
            "StyleGAN",
        )
    else:
        DAModel = getattr(
            importlib.import_module(
                name=os.path.join(MODELS_DIR, args.model).replace("/", ".")
            ),
            "DAModel",
        )
        assert issubclass(DAModel, AbstractDAModel)
    da_args, extra_args = DAModel.extra_parser(extra_args)

    print("model arguments =")
    for arg in np.sort(list(vars(da_args).keys())):
        print("\t" + arg + ":", getattr(da_args, arg))
    print("=" * LINE_LENGTH)
    if len(extra_args):
        print("Unused args: {}".format(" ".join(extra_args)))

    # CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.cuda.empty_cache()
        device = "cuda"
    else:
        print("WARNING: [CUDA unavailable] Using CPU instead!")
        device = "cpu"
    print(f"Sanity checks:")
    print(f"1. Using cuda? {torch.cuda.is_available()}")
    print(f"2. Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    return DAModel, args, da_args, extra_args, device, parser


def get_n_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])
