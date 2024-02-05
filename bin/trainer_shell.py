import os
import time
import argparse

from source.dataloader.patch_dataset import PatchDataset

from source.loggers.exp_logger import MultiLogger
from source.loggers.ml_logger import MLLogger
from source.utils.metrics import Metrics
from source.utils.misc import read_args
from source.utils.optimizers import OPTIMIZERS
from source.utils.constants import ALL_DOMAINS, MAP_DATASET_STATS, MODELS_DIR

from source.networks.resnet_model import MODEL_ARCH, ResnetModel

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


def get_model_names():
    models_name = [
        file.split(".")[0] for file in os.listdir(MODELS_DIR) if "__" not in file
    ]
    assert len(models_name) > 0, "No models found"
    return models_name


def get_opt_names():
    return [opt.name for opt in OPTIMIZERS]


def train(argv=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description="TRAINER")
    # generic args
    parser.add_argument("--gpu", type=int, default=0, help="GPU (default=%(default)s)")
    parser.add_argument(
        "--results-path",
        type=str,
        default="../results",
        help="Results path (default=%(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed (default=%(default)s)"
    )
    parser.add_argument(
        "--log",
        default=[],
        type=str,
        choices=["disk", "tensorboard"],
        help="Loggers used (disk and/or tensorboard) (default=%(default)s)",
        nargs="*",
        metavar="LOGGER",
    )
    parser.add_argument(
        "--metrics",
        default=[],
        type=str,
        choices=[metric.name for metric in Metrics.METRICS],
        help="metric used (default=%(default)s)",
        nargs="*",
        metavar="METRICS",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save trained models (default=%(default)s)",
    )
    # dataset args
    parser.add_argument(
        "--src-dom",
        type=str,
        required=True,
        nargs="*",
        help=f"Source domains, select from: {ALL_DOMAINS}",
    )
    parser.add_argument(
        "--tar-dom",
        type=str,
        default=["kather16", "crctp"],
        help="Task as list of initials",
        nargs="*",
    )
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--base-path", type=str, required=True)
    parser.add_argument(
        "--not-balanced",
        action="store_true",
        help="If not to force a balanced sampling of dataset",
    )
    parser.add_argument(
        "--norm-stats",
        default="imagenet",
        type=str,
        choices=list(MAP_DATASET_STATS.keys()),
        help="normalizing stats used to normalize input patches",
    )
    # training params
    parser.add_argument(
        "--backbone",
        required=True,
        type=str,
        choices=list(MODEL_ARCH.keys()),
        help="Learning backbone used (default=%(default)s)",
        metavar="backbone",
    )
    parser.add_argument(
        "--unfreeze-layer",
        default=10,
        type=int,
        help="last layer to freeze, 0 means freeze all",
    )
    parser.add_argument(
        "--source-model",
        default="",
        type=str,
        help="path from a pretrained source model",
    )
    parser.add_argument(
        "--gan-path",
        default="/dccstor/cpath_data/results/kth/debug/histo_da/ema_stage_0_250K_d_lr_3e-4_g_lr_2e-4_r1_2/dann_generative_style/",
        type=str,
        help="path from a pretrained source model",
    )
    parser.add_argument(
        "--gan-ckpt-n-iter",
        default=250,
        type=int,
        help="Number of ***K (thousands) iterations of the GAN checkpoint",
    )
    parser.add_argument(
        "--optimizer",
        default="radam",
        type=str,
        choices=get_opt_names(),
        help="Learning optimizer used (default=%(default)s)",
        metavar="opt",
    )
    parser.add_argument(
        "--nepochs",
        default=200,
        type=int,
        required=False,
        help="Number of epochs per training session (default=%(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        required=False,
        help="Number of samples per batch to load (default=%(default)s)",
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        required=False,
        help="Starting learning rate (default=%(default)s)",
    )
    parser.add_argument(
        "--lr-patience",
        default=20,
        type=int,
        required=False,
        help="Maximum patience to wait before decreasing learning rate (default=%(default)s)",
    )
    parser.add_argument(
        "--clipping",
        default=10.0,
        type=float,
        required=False,
        help="Clip gradient value (default=%(default)s)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        required=False,
        help="Momentum factor (default=%(default)s)",
    )
    parser.add_argument(
        "--weight-decay",
        default=5e-5,
        type=float,
        required=False,
        help="Weight decay (L2 penalty) (default=%(default)s)",
    )
    parser.add_argument(
        "--warmup-nepochs",
        default=0,
        type=int,
        required=False,
        help="Number of warm-up epochs (default=%(default)s)",
    )
    parser.add_argument(
        "--warmup-lr-factor",
        default=1.0,
        type=float,
        required=False,
        help="Warm-up learning rate factor (default=%(default)s)",
    )
    parser.add_argument("--binarize", action="store_true", help="If to binarize labels")
    parser.add_argument(
        "--moving-bn",
        action="store_true",
        help="Fix batch normalization after first task (default=%(default)s)",
    )
    # model params
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        choices=get_model_names(),
        help="Learning model used (default=%(default)s)",
        metavar="model",
    )

    parser.add_argument("--reduce", default=0, type=int, help="Reduce larger datasets")

    DAModel, args, da_args, extra_args, device, parser = read_args(argv, parser)

    # Log all arguments
    model_name = args.model
    logger = MultiLogger(
        args.results_path, model_name, loggers=args.log, save_models=False
    )
    logger.log_tag("method", model_name)
    logger.log_args(argparse.Namespace(**args.__dict__, **da_args.__dict__))
    tb_logger = MLLogger(mode="tensorboard", log_dir=logger.exp_path)

    # Collect data
    patch_whole_dataset = PatchDataset(
        original_csv=args.labels,
        split_csv=args.datasets,
        reduce=args.reduce,
        base_path=args.base_path,
    )
    n_classes = patch_whole_dataset.get_classes()
    src_dom = "_".join(args.src_dom)

    # Network instances
    base_kwargs = dict(
        nepochs=args.nepochs,
        lr=args.lr,
        src_doms=args.src_dom,
        tar_doms=args.tar_dom,
        lr_patience=args.lr_patience,
        clipgrad=args.clipping,
        momentum=args.momentum,
        wd=args.weight_decay,
        wu_nepochs=args.warmup_nepochs,
        wu_lr_factor=args.warmup_lr_factor,
        fix_bn=not args.moving_bn,
        metrics=args.metrics,
        n_classes=n_classes,
        optimizer_name=args.optimizer,
        device=device,
        unfreeze_layer=args.unfreeze_layer,
        norm_stats=args.norm_stats,
        batch_size=args.batch_size,
        not_balanced=args.not_balanced,
    )
    kwargs = {
        **base_kwargs,
        **dict(logger=logger, tb_logger=tb_logger),
        **da_args.__dict__,
    }

    # instantiate network and DA model
    base_net = ResnetModel.get_model(
        model_name=args.backbone, head_len=2, n_classes=n_classes, classifier=True
    )
    model = DAModel(base_net, patch_whole_dataset, **kwargs)
    model_save = os.path.join(args.results_path, "pretrained_models", f"{model_name}")
    if not os.path.exists(model_save):
        os.makedirs(model_save, exist_ok=True)

    # ######################
    # # Training
    # ######################
    # train on source + warmup, skip it if valid path to a model is given
    if not model.load_model(args.source_model):
        model.train_source(src_dom, warmup=True)
        model.train_source(src_dom)

    # train generative if it is not pre-loaded
    if not model.load_gan(args.gan_path, n_iter=args.gan_ckpt_n_iter):
        print("GAN not loaded -> start training")
        model.train_generative(src_dom)

    # adapt to a new model architecture saving states/dict (if needed)
    model.transform_model()

    # test and save source model if it is not pre-loaded
    model.test_source(src_dom)
    if not model.load_model(args.source_model):
        model.save_model(model_save, args.src_dom)

    # loop over target domains
    args_tar_dom = list(args.tar_dom)
    for i, tar_dom in enumerate(args_tar_dom):
        # train and test
        if i == 0:
            model.test_target(tar_dom, src_dom)
        if tar_dom == args_tar_dom[-1]:
            model.train_target(tar_dom, src_dom)
        model.test_target(tar_dom, src_dom)

        if tar_dom != args_tar_dom[-1]:
            # retrain generative model to fit targets (if needed)
            model.update_generative(tar_dom)
            # accumulate previous target domains in sources (if needed)
            model.adapt_sources(tar_dom)

        if tar_dom == args_tar_dom[-1]:
            model.save_model(model_save, [*args.src_dom, *args_tar_dom[: i + 1]])

    # test sources and targets individually
    model.test_final(src_dom)

    print(f"Elapsed time = {(time.time() - tstart) / (60 * 60):.1f} h")
    tb_logger.end()
    return logger.exp_path


if __name__ == "__main__":
    train()
