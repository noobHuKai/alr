import torch
from arl.trainer import onpolicy_trainer, policy_tester
from arl.data import Collector
from arl.env import BaseEnv
from arl.learner import PPOLearner, PPOContinuousLearner
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from arl.policy import BasePolicy


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return F.softmax(self.network(x), dim=1)


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPOPolicy(BasePolicy):
    def __init__(self, cfg: dict, env: BaseEnv) -> None:
        super().__init__(cfg, env)
        self.hidden_dim = self.learn_params.get("hidden_dim", 128)
        self.actor_lr = float(self.learn_params.get("actor_lr"))
        self.critic_lr = float(self.learn_params.get("critic_lr"))

    def run(self) -> None:
        collector = Collector(env=self.env)

        if self.action_type.is_discrete():
            self.get_discrete_learner()
        elif self.action_type.is_continuous():
            self.get_continuous_learner()

        logger = SummaryWriter(self.run_name)

        onpolicy_trainer(
            collector=collector,
            learner=self.learner,
            train_params=self.train_params,
            show_progress=True,
            logger=logger,
        )

    def get_discrete_learner(self) -> None:
        actor = PolicyNet(self.state_dim, self.hidden_dim, self.action_dim)
        critic = ValueNet(self.state_dim, self.hidden_dim)

        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.actor_lr)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.critic_lr)

        self.learner = PPOLearner(
            device=self.device,
            actor_net=actor,
            critic_net=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            learn_params=self.learn_params,
        )

    def get_continuous_learner(self) -> None:
        actor = PolicyNetContinuous(self.state_dim, self.hidden_dim, self.action_dim)
        critic = ValueNet(self.state_dim, self.hidden_dim)

        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.actor_lr)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.critic_lr)

        self.learner = PPOContinuousLearner(
            device=self.device,
            actor_net=actor,
            critic_net=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            learn_params=self.learn_params,
        )

    def run_test(self) -> None:
        collector = Collector(env=self.env)

        if self.action_type.is_discrete():
            self.get_discrete_learner()
        elif self.action_type.is_continuous():
            self.get_continuous_learner()

        self.learner.load_model(self.model_path)

        logger = SummaryWriter(self.run_name)

        policy_tester(
            collector=collector,
            learner=self.learner,
            train_params=self.train_params,
            show_progress=True,
            logger=logger,
        )
