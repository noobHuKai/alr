from arl.data import Collector, ReplayBuffer
from arl.learner import BaseLearner
from typing import Optional
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
from logging import info
import datetime


class BaseTrainer:
    def __init__(
        self,
        collector: Collector,
        train_params: dict,
        learner: BaseLearner,
        replay_buffer: Optional[ReplayBuffer] = None,
        show_progress: bool = False,
        logger=SummaryWriter,
    ) -> None:
        self.collector = collector
        self.learner = learner

        self.train_params = train_params

        self.num_episodes = train_params.get("num_episodes")

        # replay buffer
        self.replay_buffer = replay_buffer
        if self.replay_buffer is not None:
            self.buffer_minimal_size = train_params.get("buffer_minimal_size")
            self.batch_size = train_params.get("batch_size")

        # test
        self.test_n_step = train_params.get("test_n_step")

        info("train params : {}".format(train_params))

        self.is_run = False

        # tensorboard logger
        self.logger = logger
        # progress bar
        self.show_progress = show_progress
        if self.show_progress:
            self.pbar = tqdm(total=self.num_episodes)

    def policy_update(self) -> None:
        pass

    def log_update(self) -> None:
        log_dict = {}
        for log in self.log_dicts:
            for key in log:
                if key in log_dict:
                    log_dict[key] += log[key]
                else:
                    log_dict[key] = log[key]

        for key in log_dict:
            self.logger.add_scalar(key, log_dict[key], self.i_episode + 1)

    def train_episode(self) -> None:
        self.episode_transition_dict = {
            "states": [],
            "actions": [],
            "next_states": [],
            "rewards": [],
            "dones": [],
        }
        self.log_dicts = []

        self.state, _ = self.collector.env.reset()

        self.done = False
        while not self.done:
            self.train_step()

    def train_step(self) -> None:
        action = self.learner.take_action(True, self.state)
        next_state, reward, done, _ = self.collector.env.step(action)
        self.done = done

        self.episode_transition_dict["states"].append(self.state)
        self.episode_transition_dict["actions"].append(action)
        self.episode_transition_dict["next_states"].append(next_state)
        self.episode_transition_dict["rewards"].append(reward)
        self.episode_transition_dict["dones"].append(done)

        self.state = next_state

        log_dict = self.learner.get_log(True)
        log_dict["train/reward"] = reward
        self.log_dicts.append(log_dict)

    def test_episode(self) -> None:
        self.log_dicts = []

        self.state, _ = self.collector.env.reset()

        self.done = False
        while not self.done:
            self.test_step()

    def test_step(self) -> None:
        action = self.learner.take_action(False, self.state)
        next_state, reward, done, _ = self.collector.env.step(action)
        self.done = done

        self.state = next_state

        log_dict = self.learner.get_log(False)
        log_dict["test/reward"] = reward
        self.log_dicts.append(log_dict)

    def get_step_data(self):
        return (
            self.episode_transition_dict["states"][-1],
            self.episode_transition_dict["actions"][-1],
            self.episode_transition_dict["next_states"][-1],
            self.episode_transition_dict["rewards"][-1],
            self.episode_transition_dict["dones"][-1],
        )

    def run(self) -> None:
        self.is_run = True

        start_time = time.time()
        for i_episode in range(self.num_episodes):
            self.i_episode = i_episode

            self.train_episode()

            if self.test_n_step is not None:
                if (i_episode + 1) % self.test_n_step == 0:
                    self.test_episode()

            self.log_update()

            if self.show_progress:
                self.pbar.update(1)

        end_time = time.time()
        run_time = end_time - start_time
        info("train time : {}".format(datetime.timedelta(seconds=run_time)))


class BaseTester:
    def __init__(
        self,
        collector: Collector,
        learner: BaseLearner,
        train_params: dict,
        show_progress: bool = False,
        logger=SummaryWriter,
    ) -> None:
        self.collector = collector
        self.learner = learner
        self.train_params = train_params

        self.test_num_episodes = train_params.get("test_num_episodes")

        self.is_run = False

        # tensorboard logger
        self.logger = logger
        # progress bar
        self.show_progress = show_progress
        if self.show_progress:
            self.pbar = tqdm(total=self.test_num_episodes)

    def test_episode(self) -> None:
        self.log_dicts = []

        self.state, _ = self.collector.env.reset()

        self.done = False

        episode_reward = 0
        while not self.done:
            episode_reward += self.test_step()

        self.logger.add_scalar("test_model/reward", episode_reward, self.i_episode + 1)

    def test_step(self) -> float:
        action = self.learner.take_action(False, self.state)
        next_state, reward, done, _ = self.collector.env.step(action)
        self.done = done

        self.state = next_state

        return float(reward)

    def run(self) -> None:
        self.is_run = True

        for i_episode in range(self.test_num_episodes):
            self.i_episode = i_episode

            self.test_episode()

            if self.show_progress:
                self.pbar.update(1)


def policy_tester(*args, **kwargs) -> None:
    return BaseTester(*args, **kwargs).run()
