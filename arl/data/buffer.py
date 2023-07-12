import collections
import random
from typing import List

import numpy as np
from typing import Union, List, Any, Tuple

# state, action, reward, next_state, done
EnvData = Tuple[
    Union[np.ndarray, List[np.ndarray]],
    Union[Any, List[Any]],
    Union[Union[int, float], List[Union[int, float]]],
    Union[np.ndarray, List[np.ndarray]],
    Union[bool, List[bool]],
]


class ReplayBuffer(object):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity

    def add(self, data: EnvData):
        pass

    # 采样
    def sample(self, batch_size: int) -> List[EnvData]:
        pass

    def size(self) -> int:
        pass


# 随机采样 replay buffer
class RandomSampleReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, data: EnvData):
        self.buffer.append(data)

    # 随机采样
    def sample(self, batch_size: int) -> List[EnvData]:
        transitions = random.sample(self.buffer, batch_size)
        data = zip(*transitions)
        return data

    def size(self) -> int:
        return len(self.buffer)


# 顺序采样
class SequenceSampleReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    # 可能是数组，也可能是具体的值
    def add(self, data: EnvData):
        self.buffer.append(data)

    def sample(self) -> EnvData:
        return self.buffer.popleft()

    def size(self) -> int:
        return len(self.buffer)


# TODO: 改成环形队列可以节省内存
# class EnvDataBuffer:
#     def __init__(self, capacity: int):
#         self.buffer = collections.deque(maxlen=capacity)

#     # 可能是数组，也可能是具体的值
#     def add(self, data: EnvData):
#         self.buffer.append(data)

#     # 只取头部第一个数据
#     def head_sample(self) -> EnvData:
#         return self.buffer.popleft()

#     # 随机采样
#     def random_sample(self, batch_size: int) -> List[EnvData]:
#         transitions = random.sample(self.buffer, batch_size)
#         data = zip(*transitions)
#         return data

#     def size(self) -> int:
#         return len(self.buffer)
