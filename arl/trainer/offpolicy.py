from typing import Optional

from torch.utils.tensorboard import SummaryWriter
from arl.data import Collector, ReplayBuffer
from arl.learner import BaseLearner
from arl.trainer import BaseTrainer


class OffPolicyTrainer(BaseTrainer):
    def __init__(
        self,
        collector: Collector,
        train_params: dict,
        learner: BaseLearner,
        replay_buffer: Optional[ReplayBuffer] = None,
        show_progress: bool = False,
        logger=None,
    ) -> None:
        super().__init__(
            collector, train_params, learner, replay_buffer, show_progress, logger
        )

    def train_step(self) -> None:
        super().train_step()
        self.policy_update()

    def policy_update(self) -> None:
        state, action, next_state, reward, done = self.get_step_data()
        self.replay_buffer.add(state, action, reward, next_state, done)
        if self.replay_buffer.size() > self.buffer_minimal_size:
            b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(self.batch_size)

            transition_dict = {
                "states": b_s,
                "actions": b_a,
                "next_states": b_ns,
                "rewards": b_r,
                "dones": b_d,
            }
            self.learner.update(transition_dict)


def offpolicy_trainer(*args, **kwargs) -> None:
    return OffPolicyTrainer(*args, **kwargs).run()
