from arl.env import BaseEnv
from arl.learner import BaseLearner
from pathlib import Path
from typing import Optional
from arl.utils import ensure_dir


class BasePolicy:
    def __init__(self, cfg: dict, env: BaseEnv) -> None:
        self.cfg = cfg

        self.learn_params: dict = cfg.get("learn_params")
        if self.learn_params is None:
            raise Exception("learn_params is empty")

        self.train_params: dict = cfg.get("train_params")
        if self.train_params is None:
            raise Exception("train_params is empty")

        self.buffer_capacity = self.train_params.get("buffer_capacity", 10000)

        self.env = env
        self.state_dim, self.action_dim = env.get_shape()
        self.state_type, self.action_type = env.get_shape_type()

        self.learn_params["state_dim"] = self.state_dim
        self.learn_params["action_dim"] = self.action_dim
        self.learn_params["state_type"] = self.state_type
        self.learn_params["action_type"] = self.action_type

        self.device: any = cfg.get("device", "cpu")

        self.run_name = cfg.get("run_name")
        if self.run_name is None:
            raise Exception("tensorboard logger name is empty")

        self.model_path = cfg.get("model_path")
        self.learner: Optional[BaseLearner] = None

    def run(self) -> None:
        pass

    def run_test(self) -> None:
        pass

    def save_model(self) -> None:
        if self.model_path is None:
            raise Exception("model path is empty")

        model_path = Path(self.model_path)
        ensure_dir(model_path.parent)

        self.learner.save_model(self.model_path)
