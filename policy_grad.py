from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from config import configs, params
from gymenv_v2 import make_multiple_env
from rollout_gen import RolloutGenerator
from policies import AttentionPolicy, RNNPolicy, DensePolicy, RandomPolicy

def policy_grad(env_config, iterations = 150, num_processes = 8, num_trajs_per_process = 1, gamma = 0.99, **policy_params):
    env = make_multiple_env(**env_config)

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

    rollout_gen = RolloutGenerator(num_processes, num_trajs_per_process, verbose = True)
    rrecord = []

    for ite in range(iterations): # todo: use tqdm here
        memory = rollout_gen.generate_trajs(env, actor, gamma)

        memory.values = (memory.values - np.mean(memory.values)) / (np.std(memory.values) + 1e-8)
        rrecord.extend(memory.rewards)
        loss = actor.train(memory)


def main():
    env_config = configs.starter_config
    policy_params = params.rand_params

    policy_grad(env_config,                 # environment configuration
                iterations=150,             # number of iterations to run policy gradient
                num_processes=12,            # number of processes running in parallel
                num_trajs_per_process=1,    # number of trajectories per process
                gamma = 0.99,               # discount factor
                **policy_params             # actor definition
                )


if __name__ == '__main__':
    main()
