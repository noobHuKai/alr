from typing import Dict, Any, Union, Tuple, List, Optional
import numpy as np
from enum import Enum


# Action & State Space Type
class EnvSpaceType(Enum):
    Discrete = 0
    Continuous = 1

    def is_continuous(self) -> bool:
        return self == self.Continuous

    def is_discrete(self) -> bool:
        return self == self.Discrete

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class BaseEnv:
    def __init__(
        self, env_name: str, env_params: dict = {}, seed: Optional[int] = None
    ) -> None:
        self.env_name = env_name
        self.env_params = env_params
        self.seed = seed

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        pass

    def reset(self) -> Tuple[np.ndarray, Union[dict, List[dict]]]:
        pass

    def render(self) -> np.ndarray:
        pass

    def close(self) -> None:
        pass

    def get_shape_type(self) -> Tuple[EnvSpaceType, EnvSpaceType]:
        pass

    def get_shape(self):
        pass
