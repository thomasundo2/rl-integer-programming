import matplotlib.pyplot as plt
#plt.style.use('seaborn')
import time

from config import env_configs, gen_actor_params, gen_critic_params
from logger import get_filename
from helper import plot_arr, get_reward_sums


def plot_rewards(filepaths, labels = None):
    plt.figure(figsize=(15, 10), dpi=80)
    for i, filepath in enumerate(filepaths):
        reward_sums = get_reward_sums(filepath)
        if labels == None:
            label = filepath
        else:
            label = labels[i]
        plot_arr(reward_sums, label=label, window_size=100)
    plt.legend()
    curr_time = time.strftime("%Y%m%d-%H%M%S")

    plt.xlabel("trajectory")
    plt.ylabel("Reward Summation")

    plt.savefig(f"figures/{curr_time}.jpeg")
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
    filepaths_25 = ["records/randomip_n25_m25/idx_0_0/actor_attention_critic_dense/20210416-161140.txt",
                "records/randomip_n25_m25/idx_0_0/actor_attention_critic_dense/20210416-111758.txt",
                 "records/randomip_n25_m25/idx_0_0/actor_attention_critic_None/20210416-140850.txt",
                 "records/randomip_n25_m25/idx_0_0/actor_dense_critic_dense/20210416-150828.txt",
                 "records/randomip_n25_m25/idx_0_0/actor_dense_critic_dense/20210416-120436.txt",
                 "records/randomip_n25_m25/idx_0_0/actor_dense_critic_None/20210416-130451.txt",
                 "records/randomip_n25_m25/idx_0_0/actor_double_attention_critic_dense/20210416-034925.txt",
                 "records/randomip_n25_m25/idx_0_0/actor_attention_critic_None/20210416-171650.txt",
                 "records/randomip_n10_m20/idx_0_4/actor_dense_critic_None/20210417-010043.txt"
                ]

    filepaths = ["records/randomip_n10_m20/idx_0_0/actor_attention_critic_None/20210418-161243.txt",
                 "records/randomip_n10_m20/idx_0_0/actor_dense_critic_None/20210418-161009.txt",
                 "records/randomip_n10_m20/idx_0_0/attention/(n_10)(h_32)(lr_0.001).txt",
                 "records/randomip_n10_m20/idx_0_0/ppo_actor_dense_critic_None/20210418-184012.txt"
                 ]

    filepaths = ["records/randomip_n10_m20/idx_50_59/ppo_actor_dense_critic_None/20210418-190049.txt",
                 "records/randomip_n10_m20/idx_50_59/actor_dense_critic_None/20210418-190104.txt",
                 "records/randomip_n10_m20/idx_50_59/ppo_actor_dense_critic_None/20210418-220638.txt",
                 "records/randomip_n10_m20/idx_50_59/actor_dense_critic_None/20210418-220618.txt"
                 ]

    """filepaths = ["records/randomip_n25_m25/idx_0_0/actor_dense_critic_None/20210418-222700.txt", # with rnd
                 "records/randomip_n25_m25/idx_0_0/actor_dense_critic_None/20210418-222937.txt",
                 "records/randomip_n25_m25/idx_0_0/ppo_actor_dense_critic_None/20210418-222154.txt", # ppo with exploration
                 "records/randomip_n25_m25/idx_0_0/actor_dense_critic_dense/20210416-120436.txt",
                 "records/randomip_n25_m25/idx_0_0/actor_dense_critic_None/20210416-185748.txt"
                 ]"""

    """
    STARTER CONFIG 2, WILL TEST FOR THE BEST ALGORITHMS!!!!
    
    filepaths = ["records/randomip_n15_m15/idx_0_4/actor_random_critic_None_rnd_None/20210418-232807.txt",
                 "records/randomip_n15_m15/idx_0_4/ppo_actor_dense_critic_None_rnd_None/20210419-001441.txt",
                 "records/randomip_n15_m15/idx_0_4/actor_dense_critic_None_rnd_None/20210419-010446.txt",
                 "records/randomip_n15_m15/idx_0_4/ppo_actor_dense_critic_None_rnd_dense/20210419-023130.txt",
                 "records/randomip_n15_m15/idx_0_4/actor_dense_critic_None_rnd_dense/20210419-024402.txt",
                 "records/randomip_n15_m15/idx_0_4/ppo_actor_dense_critic_dense_rnd_dense/20210419-024605.txt",
                 "records/randomip_n15_m15/idx_0_4/actor_dense_critic_dense_rnd_dense/20210419-024659.txt",
                 "records/randomip_n15_m15/idx_0_4/actor_attention_critic_None_rnd_dense/20210419-024512.txt",
                 "records/randomip_n15_m15/idx_0_4/ppo_actor_dense_critic_dense_rnd_None/20210419-133431.txt",
                 "records/randomip_n15_m15/idx_0_4/ppo_actor_dense_critic_dense_rnd_dense/20210419-134535.txt",
                 "records/randomip_n15_m15/idx_0_4/ppo_actor_dense_critic_None_rnd_None/20210419-164707.txt",
                 "records/randomip_n15_m15/idx_0_4/ppo_actor_dense_critic_None_rnd_dense/20210419-164659.txt"
                 ]
    # labels corresponds to the filepath
    labels = ["Random Policy",
              "PPO (Dense Network, No Critic)",
              "Policy Grad (Dense Network, No Critic)",
              "PPO w RND(Dense Network, No Critic)",
              "Policy Grad w RND (Dense Network, No Critic)",
              "PPO w RND(Dense Network, Dense Critic)",
              "Policy Grad w RND (Dense Network, Dense Critic)",
              "Policy Grad w RND (Attention Network, No Critic)",
              "PPO (Dense Network, Dense Critic)",
              "PPO w RND (Dense Network, Dense Critic More Iterations)",
              "PPO (Dense Network, No Critic, More Iterations)",
              "PPO w RND (Dense Network, No Critic, More Iterations)"
              ]
              """
    plot_rewards(filepaths, labels)



if __name__ == '__main__':
    main()



