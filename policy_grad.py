from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# import wandb
# wandb.login()
# run=wandb.init(project="finalproject", entity="ieor-4575", tags=["n=10,m=20,i=1"])


from config import configs, params
from gymenv_v2 import make_multiple_env
from rollout_gen import RolloutGenerator
from policies import AttentionPolicy, RNNPolicy, DensePolicy, RandomPolicy
from helper import plot_arr


def build_actor(policy_params):
    """returns the type of model within policy_params
    """
    # determine the type of model
    if policy_params['model'] == 'dense':
        actor = DensePolicy(**policy_params['model_params'])
    elif policy_params['model'] == 'rnn':
        actor = RNNPolicy(**policy_params['model_params'])
    elif policy_params['model'] == 'attention':
        actor = AttentionPolicy(**policy_params['model_params'])
    elif policy_params['model'] == 'random':
        actor = RandomPolicy(**policy_params['model_params'])
    else:
        raise NotImplementedError

    return actor

def policy_grad(env_config,
                policy_params,
                iterations = 150,
                num_processes = 8,
                num_trajs_per_process = 1,
                gamma = 0.99
                ):
    env = make_multiple_env(**env_config)

    actor = build_actor(policy_params)

    rollout_gen = RolloutGenerator(num_processes, num_trajs_per_process, verbose = False)
    rrecord = []
    lossrecord = []

    for _ in tqdm(range(iterations)): # todo: use tqdm here
        memory = rollout_gen.generate_trajs(env, actor, gamma)

        memory.values = (memory.values - np.mean(memory.values)) / (np.std(memory.values) + 1e-8)
        rrecord.extend(memory.reward_sums)
        loss = actor.train(memory)
        lossrecord.append(loss)

    plot_arr(rrecord, label = "Moving Avg Reward " + policy_params['model'], window_size = 101)

    plot_arr(lossrecord, label = "Loss " + policy_params['model'], window_size = 1)


def main():
    env_config = configs.starter_config
    policy_params = params.attention_params

    policy_grad(env_config,                 # environment configuration
                iterations=350,             # number of iterations to run policy gradient
                num_processes=12,           # number of processes running in parallel
                num_trajs_per_process=1,    # number of trajectories per process
                gamma = 0.99,               # discount factor
                **policy_params             # actor definition
                )


if __name__ == '__main__':
    main()
