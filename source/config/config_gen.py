import importlib
import json
import os
from typing import Any, ChainMap, Dict, List
from dataclasses import dataclass, fields, field

from itertools import product

from source.utils.constants import MODELS_DIR


def filter_args(model, args):
    # get model in exam and filter kwargs based on its attributes
    try:
        obj_model = getattr(
            importlib.import_module(
                name=os.path.join(MODELS_DIR, model).replace("/", ".")
            ),
            "DAModel",
        )(None, None, None, None, None, None)
        filter_args = {k: v for k, v in args.items() if hasattr(obj_model, k)}
    except AttributeError as e:
        # do it for stylegan training
        print(f"While loading get: {e}")
        filter_args = args.copy()
        filter_args.pop("lamb")

    filter_args.update({"model": model})
    return filter_args


def unroll_lr_prop(keys, values):
    dict_ = {}
    for key, val in dict(zip(keys, values)).items():
        if isinstance(val, dict):
            dict_.update({"lr-" + k: v for k, v in val.items()})
        elif isinstance(val, bool):
            if val:
                dict_[key] = ""
        else:
            dict_[key] = val

    return dict_


@dataclass
class GenericParams:
    gpu: int
    results_path: str
    log: List[str]
    seed: List[int]
    datasets: str
    labels: str
    base_path: str
    metrics: List[str]

    def flatten(self) -> List[Dict[str, Any]]:
        dicts = [
            {
                "gpu": str(self.gpu),
                "results-path": self.results_path,
                "log": " ".join([log for log in self.log]),
                "seed": seed,
                "datasets": self.datasets,
                "labels": self.labels,
                "metrics": " ".join([metric for metric in self.metrics]),
                "base_path": self.base_path,
            }
            for seed in self.seed
        ]
        return dicts

    @classmethod
    def deserialize(cls, dict_: Dict[str, Any]):
        return GenericParams(**dict_)


@dataclass
class ModelParams:
    model: List[str]
    gan_batchsize: List[int]
    lamb: List[float]
    delt: List[float]
    stage: int
    disc_lr: List[float]
    gen_lr: List[float]
    gan_epochs: List[int] = field(default_factory=lambda: [280])
    gan_disc_lr: List[float] = field(default_factory=lambda: [25e-5])
    gan_gen_lr: List[float] = field(default_factory=lambda: [25e-5])
    r1_gamma: List[float] = field(default_factory=lambda: [1.0])
    attention: List[bool] = field(default_factory=lambda: [False])
    residual: List[bool] = field(default_factory=lambda: [False])

    def flatten(self) -> List[Dict[str, Any]]:
        fixed_args = {
            name: val
            for name, val in self.__dict__.items()
            if not isinstance(val, list)
        }
        sweeping_args = {
            name: val for name, val in self.__dict__.items() if isinstance(val, list)
        }
        keys, values = zip(*sweeping_args.items())
        model_combo = [
            {**fixed_args, **dict(zip(keys, val))} for val in product(*values)
        ]
        model_combo = [filter_args(combo.pop("model"), combo) for combo in model_combo]
        return model_combo

    @classmethod
    def deserialize(cls, dict_: Dict[str, Any]):
        return ModelParams(**dict_)


@dataclass
class TrainingParams:
    nepochs: int
    source_model: str
    gan_path: str
    gan_ckpt_n_iter: int
    reduce: List[int]
    batch_size: List[int]
    lr: List[float]
    lr_patience: List[float]
    optimizer: List[str]
    weight_decay: List[float]
    backbone: List[str]
    warmup_nepochs: List[int]
    warmup_lr_factor: List[float]
    unfreeze_layer: List[int]

    def flatten(self) -> List[Dict[str, Any]]:
        fixed_args = {
            name: val
            for name, val in self.__dict__.items()
            if not isinstance(val, list)
        }
        sweeping_args = {
            name: val for name, val in self.__dict__.items() if isinstance(val, list)
        }
        keys, values = zip(*sweeping_args.items())
        model_combo = [
            {**fixed_args, **dict(zip(keys, val))} for val in product(*values)
        ]
        return model_combo

    @classmethod
    def deserialize(cls, dict_: Dict[str, Any]):
        return TrainingParams(**dict_)


@dataclass
class DataParams:
    not_balanced: bool
    norm_stats: str
    src_dom: List[List[str]]
    tar_dom: List[List[str]]

    def flatten(self) -> List[Dict[str, Any]]:
        dict_list_sweeping = {
            name: val for name, val in self.__dict__.items() if isinstance(val, list)
        }
        dict_list_fixed = {
            name: [val for _ in range(len(self.src_dom))]
            for name, val in self.__dict__.items()
            if not isinstance(val, list)
        }
        dict_list = {**dict_list_sweeping, **dict_list_fixed}
        model_combo = [dict(zip(dict_list, t)) for t in zip(*dict_list.values())]
        for combo in model_combo:
            combo["src_dom"] = " ".join([str(x) for x in combo["src_dom"]])
            combo["tar_dom"] = " ".join([str(x) for x in combo["tar_dom"]])
        return model_combo

    @classmethod
    def deserialize(cls, dict_: Dict[str, Any]):
        return DataParams(**dict_)


@dataclass
class Config:
    model_params: ModelParams
    generic_params: GenericParams
    training_params: TrainingParams
    data_params: DataParams

    def flatten(self) -> List[Dict[str, Any]]:
        list_jsonstr = [
            json.dumps(dict(ChainMap(*combo)), sort_keys=True)
            for combo in product(*[val.flatten() for val in self.__dict__.values()])
        ]
        return [json.loads(jsonstr) for jsonstr in list(set(list_jsonstr))]

    @classmethod
    def deserialize(cls, dict_: Dict[str, Dict[str, Any]]):
        map_ = {field.name: field.type for field in fields(cls)}
        deser_dict = {}
        for k, v in dict_.items():
            deser_dict[k] = map_[k].deserialize(v)

        return Config(**deser_dict)


class ConfigGenerator:
    def __init__(
        self, json_file: str, lsf_queue: str = "x86_24h", cores: int = 32, mem: int = 40
    ) -> None:
        self.filename: str = json_file
        self.queue: str = lsf_queue
        self.cores: int = cores
        self.mem: int = mem
        self.param_grid: Dict[str, Any] = {}

    def load(self) -> None:
        assert self.filename.split(".")[-1] == "json", "config file not .json"
        with open(self.filename) as f:
            self.param_grid = json.load(f)

    def clean(self, list_configs: List[Dict[str, Any]]):
        list_cmds = []
        for config in list_configs:
            list_cmds.append(
                [
                    (
                        f"--{key.replace('_', '-')} {val}"
                        if not isinstance(val, bool)
                        else f"--{key.replace('_', '-')}" if val else ""
                    )
                    for key, val in config.items()
                ]
            )
        return list_cmds

    def create_lsf(self):
        list_configs = Config.deserialize(self.param_grid).flatten()
        list_configs = self.clean(list_configs.copy())
        cmds = [
            " ".join(["python ./bin/trainer_shell.py", *config])
            for config in list_configs
        ]
        for cmd in cmds:
            os.system(cmd)

    def run(self):
        self.load()
        self.create_lsf()
