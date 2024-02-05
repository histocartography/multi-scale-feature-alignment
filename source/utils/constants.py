LINE_LENGTH = 110
ALL_DOMAINS = ("kather16", "kather19", "crctp")
MODELS_DIR = "source/da_models/"
SET_TYPES = ("train", "val", "test")
HISTO_DATASETS_STATS = {
    "mean": (0.7816, 0.6039, 0.7723),
    "std": (0.1618, 0.2144, 0.1527),
}
IMAGENET_STATS = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
STD_STATS = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
MAP_DATASET_STATS = {
    "imagenet": IMAGENET_STATS,
    "histo": HISTO_DATASETS_STATS,
    "none": STD_STATS,
}
MIN_EPOCHS = 5
ITER_EPOCHS = 50
ITER_OURS = 10
ITER_GAN_EPOCHS = 1000
MAX_ITERS = 100000
STAGE_DIMS = {0: (3, 128, 128), 3: (128, 16, 16), 4: (256, 8, 8)}
DOMS = 2
