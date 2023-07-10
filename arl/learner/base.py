from typing import Dict, Any, Union
from logging import info


class BaseLearner:
    def __init__(
        self, learn_name: str, learn_params: dict, device: Union[str, int] = "cpu"
    ) -> None:
        self.learn_name = learn_name
        self.learn_params = learn_params

        self.device = device

        self.state_dim = learn_params.get("state_dim")
        self.action_dim = learn_params.get("action_dim")
        # self.state_type = learn_params["state_type"]
        # self.action_type = learn_params["action_type"]

        info("learn params : {}".format(learn_params))

    def take_action(self, is_train: bool, state):
        pass

    def get_log(self, is_train: bool):
        pass

    def update(self, transition_dict: Dict[str, Any]) -> None:
        pass

    def learn(self) -> None:
        pass

    def save_model(self, file_path: str) -> None:
        info("save model to {}".format(file_path))

    def load_model(self, file_path: str) -> None:
        info("load model to {}".format(file_path))
