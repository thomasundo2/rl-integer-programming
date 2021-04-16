from gymenv_v2 import make_multiple_env
import numpy as np

from helper import get_reward_sums, moving_avg
import wandb
wandb.login()
run=wandb.init(project="finalproject", entity="ieor-4575", tags=["training-easy"])

filepath = "records/train_10_n60_m60/idx_0_9/dense/(m_60)(n_60)(t_50)(lr_0.001).txt"
reward_sums = get_reward_sums(filepath)
reward_avgs = moving_avg(reward_sums, window_size =100)
for reward_sum, reward_avg in zip(reward_sums,reward_avgs):
    wandb.log({"Training reward": reward_sum})
    wandb.log({"Training reward moving average": reward_avg})



