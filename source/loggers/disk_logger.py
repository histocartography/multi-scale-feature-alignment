import os
import sys
import json
import shutil
import torch
from datetime import datetime

from source.loggers.exp_logger import ExperimentLogger


class Logger(ExperimentLogger):
    """Characterizes a disk logger"""

    def __init__(self, log_path, exp_name, begin_time=None):
        super(Logger, self).__init__(log_path, exp_name, begin_time)
        self.begin_time_str = self.begin_time.strftime("%Y-%m-%d-%H-%M")

        # Duplicate standard outputs
        sys.stdout = FileOutputDuplicator(
            sys.stdout,
            os.path.join(self.exp_path, f"{self.begin_time_str}-stdout.txt"),
            "w",
        )
        sys.stderr = FileOutputDuplicator(
            sys.stderr,
            os.path.join(self.exp_path, f"{self.begin_time_str}-stderr.txt"),
            "w",
        )

        # Raw log file
        self.raw_log_file = open(
            os.path.join(self.exp_path, f"{self.begin_time_str}-raw_log.txt"), "a"
        )

    def log_scalar(
        self, task, iter, name, value, group=None, curtime=None, target_task=None
    ):
        if curtime is None:
            curtime = datetime.now()

        # Raw dump
        entry = {
            "task": task,
            "iter": iter,
            "name": name,
            "value": str(value),
            "group": group,
            "time": curtime.strftime("%Y-%m-%d-%H-%M"),
        }
        if target_task:
            entry.update({"test on": target_task})
        self.raw_log_file.write(json.dumps(entry, sort_keys=True) + "\n")
        self.raw_log_file.flush()

    def log_args(self, args):
        with open(os.path.join(self.exp_path, f"{self.begin_time_str}-args"), "w") as f:
            json.dump(args.__dict__, f, separators=(",\n", " : "), sort_keys=True)

    def log_result(self, array, name, step):
        pass

    def log_figure(self, name, iter, figure, curtime=None):
        curtime = datetime.now()
        figure.savefig(
            os.path.join(
                self.exp_path,
                f"{curtime.strftime('%Y-%m-%d-%H-%M-%S')}-{name}_{iter}.png",
            )
        )
        figure.savefig(
            os.path.join(
                self.exp_path,
                f"{curtime.strftime('%Y-%m-%d-%H-%M-%S')}-{name}_{iter}.pdf",
            )
        )

    def log_final(self, general_args, model_args, results, iter=None):
        final_log = {
            "general_args": json.loads(
                json.dumps(
                    general_args.__dict__, separators=(",\n", " : "), sort_keys=True
                )
            ),
            "model_args": json.loads(
                json.dumps(
                    model_args.__dict__, separators=(",\n", " : "), sort_keys=True
                )
            ),
            "results": json.loads(
                json.dumps(results, separators=(",\n", " : "), sort_keys=True)
            ),
        }
        with open(
            os.path.join(self.exp_path, f"{self.begin_time_str}-final_results.json"),
            "w",
        ) as f:
            json.dump(final_log, f)

    def log_tag(self, tag, val):
        pass

    def log_artifact(self, path: str):
        shutil.copy2(path, os.path.join(self.exp_path, path.split("/")[-1]))

    def save_model(self, state_dict, task):
        torch.save(
            state_dict,
            os.path.join(
                self.exp_path, "models", f"task_{task}_{self.begin_time}.ckpt"
            ),
        )

    def __del__(self):
        self.raw_log_file.close()


class FileOutputDuplicator(object):
    def __init__(self, duplicate, fname, mode):
        self.file = open(fname, mode)
        self.duplicate = duplicate

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.duplicate.write(data)

    def flush(self):
        self.file.flush()
