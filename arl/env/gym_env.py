from arl.env import BaseEnv, EnvSpaceType
import gymnasium as gym
from typing import Dict, Any, List, Tuple, Union, Optional
import numpy as np


class GymEnv(BaseEnv):
    def __init__(
        self, env_name: str, env_params: dict = {}, seed: Optional[int] = None
    ) -> None:
        super().__init__(env_name, env_params, seed)
        self.env = gym.make(self.env_name, **self.env_params)

        self.action_dim = None
        self.state_dim = None

        self.get_shape()

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        state, reward, terminated, truncated, info = self.env.step(action)

        done = False
        if terminated or truncated:
            done = True

        return state, float(reward), done, info

    def reset(self) -> Tuple[np.ndarray, Union[dict, List[dict]]]:
        return self.env.reset(seed=self.seed)

    def render(self) -> np.ndarray:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def get_shape_type(self) -> Tuple[EnvSpaceType, EnvSpaceType]:
        if type(self.env.action_space) == gym.spaces.Discrete:
            self.action_type = EnvSpaceType.Discrete
        elif type(self.env.action_space) == gym.spaces.Box:
            self.action_type = EnvSpaceType.Continuous

        if type(self.env.observation_space) == gym.spaces.Discrete:
            self.state_type = EnvSpaceType.Discrete
        elif type(self.env.observation_space) == gym.spaces.Box:
            self.state_type = EnvSpaceType.Continuous

        return self.state_type, self.action_type

    def get_shape(self) -> Tuple[np.ndarray,np.ndarray]:
        self.get_shape_type()

        if self.state_dim is None:
            if self.state_type.is_discrete():
                # state is discrete type
                self.state_dim = np.array([self.env.observation_space.n])
            elif self.state_type.is_continuous():
                # state is continuous type
                # array
                self.state_dim = self.env.observation_space.shape
        if self.action_dim is None:
            if self.action_type.is_discrete():
                # action is discrete type
                self.action_dim = np.array([self.env.action_space.n])                
            elif self.action_type.is_continuous():
                # action is continuous type
                self.action_dim = self.env.action_space.shape

        return self.state_dim, self.action_dim