from source.loggers.exp_logger import ExperimentLogger
from torch.utils import tensorboard as tb


class Logger(ExperimentLogger):
    """Characterizes mlflow logger"""

    def __init__(self, log_path, exp_name, begin_time=None):
        super(Logger, self).__init__(log_path, exp_name, begin_time)
        self.tb_logger = tb.SummaryWriter(log_dir=log_path)

    def log_scalar(
        self, task, iter, name, value, group=None, curtime=None, target_task=None
    ):
        key = f"{name}"
        if task:
            key += f" {task}"
        if group:
            key += f" {group}"
        if target_task:
            key += f"_on_{target_task}"
        self.tb_logger.add_scalar(key, value, iter)

    def log_args(self, args):
        pass

    def log_final(self, general_args, model_args, results, iter=None):
        pass

    def log_tag(self, tag, val):
        pass

    def log_artifact(self, path: str):
        pass

    def save_model(self, state_dict, task):
        pass
