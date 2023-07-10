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

    def get_shape(self):
        self.get_shape_type()

        if self.state_dim is None:
            if self.state_type.is_discrete():
                # state is discrete type
                self.state_dim = self.env.observation_space.n
            elif self.state_type.is_continuous():
                # state is continuous type
                # array
                self.state_dim = self.env.observation_space.shape
        if self.action_dim is None:
            if self.action_type.is_discrete():
                # action is discrete type
                self.action_dim = self.env.action_space.n
            elif self.action_type.is_continuous():
                # action is continuous type
                self.action_dim = self.env.action_space.shape[0]

        return self.state_dim, self.action_dim


class GymContinuousToDiscreteEnv(GymEnv):
    def __init__(
        self,
        env_name: str,
        action_dim: int,
        env_params: dict = {},
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(env_name, env_params, seed)

        if self.action_type != EnvSpaceType.Continuous:
            raise Exception("action space must continuous")

        self.get_action_space_bound()
        self.action_dim = action_dim

    def get_action_space_bound(self):
        # continuous action min value
        self.action_lowbound = self.env.action_space.low[0]
        # continuous action max value
        self.action_upbound = self.env.action_space.high[0]
        return self.action_lowbound, self.action_upbound

    # aciton discrete to continuous
    def dis_to_con(self, discrete_action):
        return self.action_lowbound + (discrete_action / (self.action_dim - 1)) * (
            self.action_upbound - self.action_lowbound
        )

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action_continuous = self.dis_to_con(action)
        state, reward, terminated, truncated, info = self.env.step([action_continuous])

        done = False
        if terminated or truncated:
            done = True

        return state, float(reward), done, info
