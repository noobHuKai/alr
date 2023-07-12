from ..collector import Collector
from ..learner import BaseLearner
from . import BaseTrainer


class OffPolicyTrainer(BaseTrainer):
    def __init__(
        self,
        collector: Collector,
        train_params: dict,
        learner: BaseLearner,
        show_progress: bool = False,
        logger=...,
    ) -> None:
        super().__init__(collector, train_params, learner, show_progress, logger)
        agent = self.learner
        self.collector.policy_update(agent)

    # offpolicy 用 copy
    # def train_episode(self) -> None:
    #     self.log_dicts = []
    #     agent = self.learner
    #     rewards = self.collector.step_collect(agent)

    #     for reward in rewards:
    #         # 随机采样的数据，不是一局的数据
    #         # 浅拷贝，策略训练是，环境的策略也会更新
    #         self.policy_update()

    #         log_dict = self.learner.get_log(True)
    #         log_dict["train/reward"] = reward
    #         self.log_dicts.append(log_dict)
    def train_episode(self) -> None:
        self.log_dicts = []
        total_reward = self.collector.step_collect()

        self.policy_update()

        agent = self.learner
        self.collector.policy_update(agent)

        while not self.collector.done:
            reward = self.collector.step_collect()
            total_reward += reward
            self.policy_update()

            agent = self.learner
            self.collector.policy_update(agent)

        log_dict = self.learner.get_log(True)
        log_dict["train/reward"] = total_reward
        self.log_dicts.append(log_dict)

    def policy_update(self) -> None:
        buffer = self.collector.get_buffer()
        if buffer.size() > self.buffer_minimal_size:
            env_data = buffer.sample(self.batch_size)
            self.learner.update(env_data)


def offpolicy_trainer(*args, **kwargs) -> None:
    return OffPolicyTrainer(*args, **kwargs).run()
