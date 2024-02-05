import importlib
import time
from http.client import RemoteDisconnected


class MLLogger:

    def __init__(self, mode, log_dir=None):

        self.mode = mode
        self.log_dir = log_dir
        self.tb_logger = None
        self.mlflow_module = None
        if self.mode in ("tensorboard", "both"):
            tb_module = importlib.import_module("torch.utils.tensorboard")
            self.tb_logger = getattr(tb_module, "SummaryWriter")(log_dir=self.log_dir)

        if self.mode in ("mlflow", "both"):
            self.mlflow_module = importlib.__import__("mlflow")

    def start(self, params):
        if self.mlflow_module is not None:
            self.mlflow_call(func=self.mlflow_module.start_run)
            self.mlflow_call(func=self.mlflow_module.log_params, params=params)

    def end(self):
        if self.mlflow_module is not None:
            self.mlflow_call(func=self.mlflow_module.end_run)

        if self.tb_logger is not None:
            self.tb_logger.flush()
            self.tb_logger.close()

    def run(self, func_name, *args, mode="tb", **kwargs):

        assert mode in ("tb", "mlflow")

        if mode == "tb" and self.tb_logger is not None:
            if func_name == "log_scalars":
                return self.tb_log_scalars(*args, **kwargs)
            else:
                tb_log_func = getattr(self.tb_logger, func_name)
                return tb_log_func(*args, **kwargs)

        if mode == "mlflow" and self.mlflow_module is not None:
            func = getattr(self.mlflow_module, func_name)
            return self.mlflow_call(func, *args, **kwargs)

        return None

    def mlflow_call(self, func, *args, attempts=7, delay=5, **kwargs):

        if func == self.mlflow_module.log_params:
            for key, val in kwargs["params"].items():
                if len(str(val)) > 250:
                    kwargs["params"][key] = str(val)[:250]

        while attempts > 1:
            try:
                return func(*args, **kwargs)
            except RemoteDisconnected:
                print(
                    "MLFLOW remote disconnected. Trying again in {}s...".format(delay)
                )
                time.sleep(delay)
                delay *= 2
                attempts -= 1
        return func(*args, **kwargs)

    def tb_log_scalars(self, metric_dict, step):
        for k, v in metric_dict.items():
            self.tb_logger.add_scalar(k, v, step)
