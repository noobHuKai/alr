import torch
from arl.learner import BaseLearner
import numpy as np
from typing import Dict, Any, Union, Optional
import torch.nn.functional as F
from arl.utils import compute_advantage


class PPOLearner(BaseLearner):
    def __init__(
        self,
        learn_params: dict,
        device: Union[str, int],
        actor_net: Optional[torch.nn.Module] = None,
        critic_net: Optional[torch.nn.Module] = None,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        critic_optimizer: Optional[torch.optim.Optimizer] = None,
        learn_name="PPO",
    ) -> None:
        super().__init__(learn_name, learn_params, device)

        self.actor = actor_net.to(device)
        self.critic = critic_net.to(device)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        # 折扣因子
        self.gamma = self.learn_params.get("gamma")
        self.lmbda = self.learn_params.get("lmbda")
        # 一条序列的数据用来训练轮数
        self.epochs = self.learn_params.get("epochs")
        # PPO中截断范围的参数
        self.eps = self.learn_params.get("eps")

        # log
        self.actor_loss = None
        self.critic_loss = None

    def take_action(self, is_train: bool, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict: Dict[str, Any]) -> None:
        self.states = torch.tensor(
            np.array(transition_dict["states"]), dtype=torch.float
        ).to(self.device)
        self.actions = (
            torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        )
        self.rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        self.next_states = torch.tensor(
            np.array(transition_dict["next_states"]), dtype=torch.float
        ).to(self.device)
        self.dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        self.learn()

    def learn(self) -> None:
        td_target = self.rewards + self.gamma * self.critic(self.next_states) * (
            1 - self.dones
        )

        td_delta = td_target - self.critic(self.states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(
            self.device
        )
        old_log_probs = torch.log(
            self.actor(self.states).gather(1, self.actions)
        ).detach()
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(self.states).gather(1, self.actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            self.actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            self.critic_loss = torch.mean(
                F.mse_loss(self.critic(self.states), td_target.detach())
            )
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            self.actor_loss.backward()
            self.critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def get_log(self, is_train: bool):
        if is_train:
            log_dict = {
                "train/actor_loss": 0
                if self.actor_loss is None
                else self.actor_loss.item(),
                "train/critic_loss": 0
                if self.critic_loss is None
                else self.critic_loss.item(),
            }
        else:
            log_dict = {}
        return log_dict

    def save_model(self, file_path: str) -> None:
        super().save_model(file_path)
        torch.save(self.actor.state_dict(), file_path)

    def load_model(self, file_path: str) -> None:
        super().load_model(file_path)
        self.actor.load_state_dict(torch.load(file_path))


class PPOContinuousLearner(PPOLearner):
    def __init__(
        self,
        learn_params: dict,
        device: Union[str, int],
        actor_net: Optional[torch.nn.Module] = None,
        critic_net: Optional[torch.nn.Module] = None,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        critic_optimizer: Optional[torch.optim.Optimizer] = None,
        learn_name="PPOContinuous",
    ) -> None:
        super().__init__(
            learn_params,
            device,
            actor_net,
            critic_net,
            actor_optimizer,
            critic_optimizer,
            learn_name,
        )

    def take_action(self, is_train: bool, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return [action.item()]

    def learn(self) -> None:
        td_target = self.rewards + self.gamma * self.critic(self.next_states) * (
            1 - self.dones
        )

        td_delta = td_target - self.critic(self.states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(
            self.device
        )

        mu, std = self.actor(self.states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(self.actions)
        for _ in range(self.epochs):
            mu, std = self.actor(self.states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(self.actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(self.states), td_target.detach())
            )
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
