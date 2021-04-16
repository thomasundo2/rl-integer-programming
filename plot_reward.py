from config import env_configs, gen_actor_params, gen_critic_params
from logger import get_filename
from helper import plot_arr
import matplotlib.pyplot as plt
def get_reward_sums(filepath):
    print(filepath)
    try:
        f = open(filepath, 'r')
    except:
        print("Could not find a historical run of the env_config and policy_params.")
        raise FileNotFoundError

    reward_sums = []
    for i, reward_sum in enumerate(f):
        if i < 3:
            continue  # parameter descriptions
        try:
            reward_sums.append(float(reward_sum))
        except ValueError:
            continue
    return reward_sums


def main():

    env_config = env_configs.starter_config
    policy_params = gen_actor_params.gen_dense_params(m=25, n=25, t=20)
    critic_params = gen_critic_params.gen_no_critic()
    file_dir, _ = get_filename(env_config, policy_params, critic_params)

    file_name = "(m_20)(n_10)(t_10)(lr_0.001).txt"
    filepath = file_dir + file_name

    reward_sums = get_reward_sums(filepath)
    label = f"Actor: {policy_params['model']}, Critic: {critic_params['model']}"
    plot_arr(reward_sums, label=label, window_size=100)

    ###################

    critic_params = gen_critic_params.gen_critic_dense(m=25, n=25, t=20)
    file_dir, _ = get_filename(env_config, policy_params, critic_params)

    file_name = "(m_20)(n_10)(t_10)(lr_0.001).txt"
    filepath = file_dir + file_name
    reward_sums = get_reward_sums(filepath)
    label = f"Actor: {policy_params['model']}, Critic: {critic_params['model']}"
    plot_arr(reward_sums, label=label, window_size=100)


    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()



