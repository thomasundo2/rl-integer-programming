from config import configs, params
from logger import get_filename
from helper import plot_arr

def get_reward_sums(filepath):
    try:
        f = open(filepath, 'r')
    except:
        print("Could not find a historical run of the env_config and policy_params.")
        raise FileNotFoundError

    reward_sums = []
    for i, reward_sum in enumerate(f):
        if i == 0:
            continue  # hyperparameter line
        try:
            reward_sums.append(float(reward_sum))
        except ValueError:
            continue
    return reward_sums


def main():
    env_config = configs.easy_config
    # policy_params = params.gen_attention_params(n=60, h = 32)
    policy_params = params.gen_dense_params(m=60, n=60, t=50)

    file_dir, file_name = get_filename(env_config, policy_params)
    filepath = file_dir + file_name

    reward_sums = get_reward_sums(filepath)
    plot_arr(reward_sums, label="Moving Avg Reward " + policy_params['model'], window_size=101)
    plot_arr(reward_sums, label="Reward " + policy_params['model'], window_size=1)
if __name__ == '__main__':
    main()



