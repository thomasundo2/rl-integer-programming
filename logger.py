import numpy as np
from pathlib import Path
import time


def get_filename(env_config, policy_params, critic_params, rnd_params, ppo_tag = False):
    start_idx = env_config['idx_list'][0]
    end_idx = env_config['idx_list'][-1]


    if ppo_tag:
        file_dir = f"records/" \
                   f"{env_config['load_dir'][10:]}/" \
                   f"idx_{start_idx}_{end_idx}/" \
                   f"ppo_actor_{policy_params['model']}_critic_{critic_params['model']}_rnd_{rnd_params['model']}/"
    else:
        file_dir = f"records/" \
                   f"{env_config['load_dir'][10:]}/" \
                   f"idx_{start_idx}_{end_idx}/" \
                   f"actor_{policy_params['model']}_critic_{critic_params['model']}_rnd_{rnd_params['model']}/"
    file_name = time.strftime("%Y%m%d-%H%M%S")
    file_name += ".txt"

    return file_dir, file_name


class RewardLogger(object):
    # todo: when actor critic are combined and ppo config is implemented, get rid of ppo tag
    def __init__(self, env_config, policy_params, critic_params, rnd_params, hyperparameters, ppo_tag = False):
        file_dir, file_name = get_filename(env_config, policy_params, critic_params, rnd_params, ppo_tag)
        Path(file_dir).mkdir(parents=True, exist_ok=True)

        self.filepath = file_dir + file_name

        with open(self.filepath, "w+") as f:
            f.write(str(policy_params) + "\n")
            f.write(str(critic_params) + "\n")
            f.write(str(rnd_params) + "\n")
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
