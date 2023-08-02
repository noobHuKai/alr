from arl.env import GymEnv
from arl.policy import DQNPolicy, BasePolicy
from arl.utils import load_yaml
import torch
from os import path, listdir
import logging
import fire
import os


class ARLPolicy:
    def get_policy(alg: str, env_name: str) -> BasePolicy:
        cfg = load_yaml(path.join("configs", alg, env_name + ".yaml"))
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        cfg["device"] = device
        cfg["env_name"] = env_name

        env_to_dis = cfg.get("env_to_dis", False)
        if env_to_dis:
            action_dim = cfg.get("action_dim").get("action_dim")
            # env = GymContinuousToDiscreteEnv(env_name=env_name, action_dim=action_dim)
        else:
            env = GymEnv(env_name=env_name)

        # dqn 实现连续是将离散动作拆成连续动作
        if alg == "dqn":
            policy = DQNPolicy(cfg, env)

        # ppo 实现连续动作是采样方式的不同
        # elif alg == "ppo":
        #     policy = PPOPolicy(cfg, env)
        return policy

    def run_all_model(run_model_func):
        algs = listdir("configs")
        for alg in algs:
            env_names = [
                path.splitext(file_name)[0]
                for file_name in listdir(path.join("configs", alg))
            ]

            for env_name in env_names:
                run_model_func(alg, env_name)

    def train_model(alg: str, env_name: str):
        policy = ARLPolicy.get_policy(alg, env_name)

        policy.run()
        policy.save_model()

    def test_model(alg: str, env_name: str):
        policy = ARLPolicy.get_policy(alg, env_name)
        policy.run_test()

    def train_all_model():
        ARLPolicy.run_all_model(ARLPolicy.train_model)

    def test_all_model():
        ARLPolicy.run_all_model(ARLPolicy.test_model)


class ARLTest(object):
    def all(self):
        logging.info("test all model")
        ARLPolicy.test_all_model()

    def model(self, alg, env_name):
        logging.info("{} test {} model".format(env_name, alg))
        ARLPolicy.test_model(alg, env_name)


class ARLTrain(object):
    def all(self):
        logging.info("train all model")
        ARLPolicy.train_all_model()

    def model(self, alg, env_name):
        logging.info("{} train {} model".format(env_name, alg))
        ARLPolicy.train_model(alg, env_name)


class ARLCli(object):
    def __init__(self):
        self.test = ARLTest()
        self.train = ARLTrain()


if __name__ == "__main__":
    # log init
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    os.environ["PAGER"] = "cat"
    fire.Fire(ARLCli)
