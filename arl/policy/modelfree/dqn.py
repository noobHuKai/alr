import torch
from arl.trainer import offpolicy_trainer, policy_tester
from arl.data import Collector, ReplayBuffer
from arl.env import BaseEnv
from arl.learner import DQNLearner
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from arl.policy import BasePolicy
import numpy as np


class QNet(torch.nn.Module):
    """只有一层隐藏层的Q网络"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.network(x)


class ConvolutionalQNet(torch.nn.Module):
    """加入卷积层的Q网络"""

    def __init__(self, action_dim, in_channels=4):
        super(ConvolutionalQNet, self).__init__()
        self.conv = nn.Sequential(
            # (W,H)
            nn.Conv2d(
                in_channels=in_channels, out_channels=32, kernel_size=8, stride=4
            ),
            # ( (W-8)/4+1, (H-8)/4+1) => (W/4-1,H/4-1)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            # ((W/4-1-4)/2+1,(H/4-1-4)/2+1) => (W/8-1.5,H/8-1.5)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            #  ((W/8-1.5-3)/1+1,((H/8-1.5-3)/1+1)) => (W/8-3.5,H/8-3.5)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            # 图片大小
            nn.Linear(in_features=64 * 22 * 16, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=action_dim),
        )

    def transform_state(self, x):
        raw_data = x.cpu().data.numpy()
        trans_data = raw_data.transpose(0, 3, 1, 2)

        x = torch.tensor(np.array(trans_data), dtype=torch.float).to(0)
        return x

    def forward(self, x):
        x = self.transform_state(x)

        conv_out = self.conv(x)
        return self.fc(conv_out)


# 调参心得：
# max_q_value 前期是不训练的，所以 reward 前期是随机值
# max_q_value 越大越好，呈增长趋势好，reward 会上升，下降趋势不行，reward 会下降
# 奖励下降的原因是灾难性遗忘 https://en.wikipedia.org/wiki/Catastrophic_interference
class DQNPolicy(BasePolicy):
    def __init__(self, cfg: dict, env: BaseEnv) -> None:
        super().__init__(cfg, env)
        self.hidden_dim = self.learn_params.get("hidden_dim")
        self.lr = self.learn_params.get("lr")

        self.set_qnet()

    def set_qnet(self) -> None:
        if len(self.state_dim) == 3:
            self.q_net = ConvolutionalQNet(
                self.action_dim, in_channels=self.state_dim[2]
            )
        else:
            self.q_net = QNet(self.state_dim[0], self.hidden_dim, self.action_dim)

    def run(self) -> None:
        collector = Collector(env=self.env)
        replay_buffer = ReplayBuffer(capacity=self.buffer_capacity)

        optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(self.lr))

        self.learner = DQNLearner(
            device=self.device,
            q_net=self.q_net,
            optimizer=optimizer,
            learn_params=self.learn_params,
        )

        logger = SummaryWriter(self.run_name)

        offpolicy_trainer(
            collector=collector,
            learner=self.learner,
            train_params=self.train_params,
            replay_buffer=replay_buffer,
            show_progress=True,
            logger=logger,
        )

    def run_test(self) -> None:
        if self.model_path is None:
            raise Exception("model path is empty")

        collector = Collector(env=self.env)

        self.learner = DQNLearner(
            device=self.device,
            q_net=self.q_net,
            learn_params=self.learn_params,
        )

        self.learner.load_model(self.model_path)

        logger = SummaryWriter(self.run_name)

        policy_tester(
            collector=collector,
            learner=self.learner,
            train_params=self.train_params,
            show_progress=True,
            logger=logger,
        )
