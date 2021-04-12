import numpy as np
from pathlib import Path


class RewardLogger(object):
    def __init__(self, env_config, policy_params, hyperparameters):
        start_idx = env_config['idx_list'][0]
        end_idx = env_config['idx_list'][-1]

        file_dir = f"records/" \
                   f"{env_config['load_dir'][10:]}/" \
                   f"idx_{start_idx}_{end_idx}/" \
                   f"{policy_params['model']}/"
        Path(file_dir).mkdir(parents=True, exist_ok=True)

        file_name = ""
        for param_name, value in policy_params['model_params'].items():
            file_name += f"({param_name}_{value})"
        file_name += ".txt"

        self.filepath = file_dir + file_name

        with open(self.filepath, "w+") as f:
            f.write(str(hyperparameters) + "\n")

        # not used but may implement plotting functionality later
        self.reward_record = []

    def record(self, reward_sums):
        """
        :param records: list of trajectory sum of rewards
        :return:
        """
        self.reward_record.extend(reward_sums)
        reward_sums_str = "\n".join(list(map(str, reward_sums)))
        with open(self.filepath, "a") as f:
            f.write(reward_sums_str)
