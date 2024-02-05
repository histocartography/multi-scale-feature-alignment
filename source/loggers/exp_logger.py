import os
import importlib
from datetime import datetime


class ExperimentLogger:
    """Main class for experiment logging"""

    def __init__(self, log_path, exp_name, begin_time=None):
        if begin_time is None:
            self.begin_time = datetime.now()
        else:
            self.begin_time = begin_time
        self.exp_name = exp_name
        self.log_path = log_path
        self.exp_path = os.path.join(self.log_path, f"{self.exp_name}")

    def log_scalar(self, task, iter, name, value, group=None, curtime=None):
        pass

    def log_args(self, args):
        pass

    def log_result(self, array, name, step):
        pass

    def log_figure(self, name, iter, figure, curtime=None):
        pass

    def log_final(self, general_args, model_args, results):
        pass

    def log_artifact(path: str):
        pass

    def save_model(self, state_dict, task):
        pass


class MultiLogger(ExperimentLogger):
    """This class allows to use multiple loggers"""

    def __init__(self, log_path, exp_name, loggers=None, save_models=True):
        super(MultiLogger, self).__init__(log_path, exp_name)
        if os.path.exists(self.exp_path):
            print("WARNING: {} already exists!".format(self.exp_path))
        else:
            os.makedirs(os.path.join(self.exp_path, "models"))

        self.save_models = save_models
        self.loggers = []
        for l in loggers:
            lclass = getattr(
                importlib.import_module(name="source.loggers." + l + "_logger"),
                "Logger",
            )
            self.loggers.append(lclass(self.log_path, self.exp_name))

    def log_scalar(
        self, task, iter, name, value, group=None, curtime=None, target_task=None
    ):
        if curtime is None:
            curtime = datetime.now()
        for l in self.loggers:
            l.log_scalar(task, iter, name, value, group, curtime, target_task)

    def log_args(self, args):
        for l in self.loggers:
            l.log_args(args)

    def log_result(self, array, name, step):
        for l in self.loggers:
            l.log_result(array, name, step)

    def log_figure(self, name, iter, figure, curtime=None):
        if curtime is None:
            curtime = datetime.now()
        for l in self.loggers:
            l.log_figure(name, iter, figure, curtime)

    def log_final(self, general_args, model_args, results, iter=None):
        for l in self.loggers:
            l.log_final(general_args, model_args, results, iter)

    def log_tag(self, tag, val):
        for l in self.loggers:
            l.log_tag(tag, val)

    def log_artifact(self, path):
        for l in self.loggers:
            l.log_artifact(path)

    def save_model(self, state_dict, task):
        if self.save_models:
            for l in self.loggers:
                l.save_model(state_dict, task)
