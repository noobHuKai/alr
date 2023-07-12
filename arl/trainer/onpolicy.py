# from typing import Optional

# from arl.data import Collector, ReplayBuffer
# from arl.learner import BaseLearner
# from arl.trainer import BaseTrainer


# class OnPolicyTrainer(BaseTrainer):
#     def __init__(
#         self,
#         collector: Collector,
#         train_params: dict,
#         learner: BaseLearner,
#         replay_buffer: Optional[ReplayBuffer] = None,
#         show_progress: bool = False,
#         logger=None,
#     ) -> None:
#         super().__init__(
#             collector, train_params, learner, replay_buffer, show_progress, logger
#         )

#     def train_episode(self) -> None:
#         super().train_episode()
#         self.policy_update()

#     def policy_update(self) -> None:
#         self.learner.update(self.episode_transition_dict)


# def onpolicy_trainer(*args, **kwargs) -> None:
#     return OnPolicyTrainer(*args, **kwargs).run()
