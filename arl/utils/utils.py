import torch
import os
import yaml
from typing import Dict
import numpy as np
import re


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


def ensure_dir(dir_path):
    # 判断路径存在
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# 读取配置文件
def load_yaml(cfg_path) -> Dict:
    with open(cfg_path, "r") as file:
        cfg = yaml.safe_load(file)
        return cfg


def to_snake_case(string, rest_to_lower: bool = False):
    string = re.sub(r"(?:(?<=[a-z])(?=[A-Z]))|[^a-zA-Z]", " ", string).replace(" ", "_")

    if rest_to_lower == True:
        return "".join(string.lower())
    else:
        return "".join(string)
