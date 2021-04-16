import matplotlib.pyplot as plt
import time

from config import env_configs, gen_actor_params, gen_critic_params
from logger import get_filename
from helper import plot_arr, get_reward_sums


def plot_rewards(filepaths):
    plt.figure(figsize=(15, 10), dpi=80)
    for filepath in filepaths:
        reward_sums = get_reward_sums(filepath)
        label = filepath
        plot_arr(reward_sums, label=label, window_size=100)
    plt.legend()
    curr_time = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"figures/{curr_time}.jpeg")
    plt.xlabel("trajectory")
    plt.ylabel("Reward Summation")
    plt.show()
def main():
    """
    env_config = env_configs.veryeasy_config
    policy_params = gen_actor_params.gen_attention_params(n=25, h=None)
    critic_params = gen_critic_params.gen_critic_dense(m=25, n=25, t=20, lr=None)
    file_dir, _ = get_filename(env_config, policy_params, critic_params)

    file_name = "20210416-025124.txt"
    filepath = file_dir + file_name
    reward_sums = get_reward_sums(filepath)
    label = f"Actor: {policy_params['model']}, Critic: {critic_params['model']}"
    plot_arr(reward_sums, label=label, window_size=100)
    """
    filepaths = ["records/randomip_n25_m25/idx_0_0/actor_attention_critic_dense/20210416-161140.txt",
                "records/randomip_n25_m25/idx_0_0/actor_attention_critic_dense/20210416-111758.txt",
                 "records/randomip_n25_m25/idx_0_0/actor_attention_critic_None/20210416-140850.txt",
                 "records/randomip_n25_m25/idx_0_0/actor_dense_critic_dense/20210416-150828.txt",
                 "records/randomip_n25_m25/idx_0_0/actor_dense_critic_dense/20210416-120436.txt",
                 "records/randomip_n25_m25/idx_0_0/actor_dense_critic_None/20210416-130451.txt",
                 "records/randomip_n25_m25/idx_0_0/actor_double_attention_critic_dense/20210416-034925.txt",
                 "records/randomip_n25_m25/idx_0_0/actor_attention_critic_None/20210416-171650.txt"
                ]

    plot_rewards(filepaths)


if __name__ == '__main__':
    main()



