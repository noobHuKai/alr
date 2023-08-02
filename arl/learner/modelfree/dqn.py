import torch
from arl.data import EnvData, ReplayBuffer
from arl.learner import BaseLearner
import numpy as np
from typing import Dict, Any, Union, Optional
import copy
import torch.nn.functional as F


class DQNLearner(BaseLearner):

    def __init__(
        self,
        learn_params: dict,
         buffer: ReplayBuffer,
        device: Union[str, int],
        q_net: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        learn_name="DQN",
    ) -> None:
        super().__init__(learn_name, learn_params,buffer, device)
        # Q网络
        self.q_net = q_net.to(device)
        # 目标网络
        self.target_q_net = copy.deepcopy(q_net).to(device)
        # 使用Adam优化器
        self.optimizer = optimizer

        # 折扣因子
        self.gamma = self.learn_params.get("gamma")
        # epsilon-贪婪策略
        self.epsilon = self.learn_params.get("epsilon")
        # 目标网络更新频率
        self.target_update = self.learn_params.get("target_update")
        # 计数器,记录更新次数
        self.count = 0
        # 采样的数据
        self.batch_size =  64

        # log
        self.dqn_loss = None
        self.max_q_value = None

    def take_action(self, is_train: bool, state):
        if not is_train and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self) -> None:
        states, actions, rewards, next_states, dones =  self.buffer.sample(self.batch_size)
     
        self.states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        self.actions = torch.tensor(actions).view(-1, 1).to(self.device)
        self.rewards = (
            torch.tensor(np.array(rewards), dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        self.next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(
            self.device
        )
        self.dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)

        self.learn()

    def learn(self) -> None:
        q_values = self.q_net(self.states)
        self.q_values = q_values.gather(1, self.actions)  # Q值
        self.max_q_value = q_values.max().item()

        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(self.next_states).max(1)[0].view(-1, 1)

        q_targets = self.rewards + self.gamma * max_next_q_values * (
            1 - self.dones
        )  # TD误差目标
        self.dqn_loss = torch.mean(F.mse_loss(self.q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def get_log(self, is_train: bool):
        if is_train:
            log_dict = {
                "train/dqn_loss": 0 if self.dqn_loss is None else self.dqn_loss.item(),
                "train/max_q_value": 0
                if self.max_q_value is None
                else self.max_q_value,
            }
        else:
            log_dict = {
                "test/max_q_value": 0 if self.max_q_value is None else self.max_q_value,
            }
        return log_dict

    def save_model(self, file_path: str) -> None:
        super().save_model(file_path)
        torch.save(self.q_net.state_dict(), file_path)

    def load_model(self, file_path: str) -> None:
        super().load_model(file_path)
        self.q_net.load_state_dict(torch.load(file_path))
