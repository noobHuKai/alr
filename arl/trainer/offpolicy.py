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




    def train_episode(self) -> None:
        self.log_dicts = []
        total_reward = 0
        step_is_done = False
    
        while not step_is_done:
            reward = self.collector.step_collect()
            total_reward += reward
            self.policy_update()

            self.collector.policy_update(self.learner)
            step_is_done = self.collector.done
                        
        log_dict = self.learner.get_log(True)
        log_dict["train/reward"] = total_reward
        self.log_dicts.append(log_dict)
    
    # def test_episode(self) -> None:
    #     self.log_dicts = []
    #     total_reward = 0
    #     step_is_done = False
    
    #     while not step_is_done:
    #         reward = self.collector.step_collect()
    #         total_reward += reward

    #         step_is_done = self.collector.done
            
    #     log_dict = self.learner.get_log(False)
    #     log_dict["test/reward"] = total_reward
    #     self.log_dicts.append(log_dict)
        
    def policy_update(self) -> None:
        buffer = self.collector.get_buffer()
        if buffer.size() > self.buffer_minimal_size:
            self.learner.update()


def offpolicy_trainer(*args, **kwargs) -> None:
    return OffPolicyTrainer(*args, **kwargs).run()
