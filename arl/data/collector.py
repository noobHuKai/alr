from arl.env import BaseEnv
from typing import Dict, Any, Tuple
import numpy as np


class Collector:
    def __init__(self, env: BaseEnv, env_num: int = 1) -> None:
        self.env = env
        self.env_num = env_num

        # 加个环形队列收集数据
        # 好处节省内存

    def policy_update_fn() -> None:
        pass

    def log_update_data() -> None:
        pass

    def test_step() -> None:
        pass

    def train_step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        return self.env.step(action)
