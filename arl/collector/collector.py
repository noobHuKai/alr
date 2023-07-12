import numpy as np
from typing import List, Union
from ..data import ReplayBuffer
from ..learner import BaseLearner
from ..data import EnvData
from ..env import BaseEnv


# 用环形队列来收集数据
# offpolicy 异策略收集一步的数据
# onpolicy 同策略收集一局的数据


# 取action
# offpolicy 无所谓
# onpolicy 用 deepcopy 来保证是同一个策略
# 有滞后性
class Collector:
    def __init__(
        self, env: BaseEnv, buffer: ReplayBuffer, transform_state_fn=None
    ) -> None:
        # env
        self.env = env
        # env data buffer
        self.buffer = buffer
        # 转换状态的函数
        self.transform_state_fn = transform_state_fn

        self.done = True

    # 转换状态
    def transform_state(self, state: np.ndarray) -> np.ndarray:
        if self.transform_state_fn is None:
            return state
        else:
            return self.transform_state_fn(state)

    def policy_update(self, agent: BaseLearner) -> None:
        self.agent = agent

    # 按步收集数据
    # def step_collect(
    #     self

    # ) -> List[Union[float, int]]:
    #     state, _ = self.env.reset()
    #     state = self.transform_state(state)

    #     rewards = []
    #     done = False
    #     while not done:
    #         action = self.agent.take_action(False, state)
    #         next_state, reward, done, _ = self.env.step(action)
    #         next_state = self.transform_state(next_state)

    #         env_data = (state, action, reward, next_state, done)
    #         self.buffer.add(env_data)

    #         state = next_state
    #         rewards.append(reward)

    #     return rewards

    def step_is_done(self):
        return self.done

    # 按步收集数据
    def step_collect(self) -> List[Union[float, int]]:
        if self.done:
            self.state, _ = self.env.reset()

        action = self.agent.take_action(False, self.state)
        next_state, reward, self.done, _ = self.env.step(action)

        env_data = (self.state, action, reward, next_state, self.done)
        self.buffer.add(env_data)

        self.state = next_state
        return reward

    # 按 局收集数据
    def episode_collect(
        self,
        agent: BaseLearner,
    ) -> int:
        states, actions, rewards, next_states, dones = [], [], [], [], []

        state, _ = self.env.reset()
        state = self.transform_state(state)

        i = 0
        done = False
        while not done:
            action = agent.take_action(state)

            next_state, reward, done, _ = self.env.step(action)
            next_state = self.transform_state(next_state)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            i += 1

        env_data = EnvData(states, actions, rewards, next_states, dones)
        self.buffer.add(env_data)
        return i

    def get_buffer(self) -> ReplayBuffer:
        return self.buffer
