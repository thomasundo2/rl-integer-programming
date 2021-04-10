from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from config import configs, params
from gymenv_v2 import make_multiple_env
from rollout_gen import RolloutGenerator
from policies import AttentionPolicy, RNNPolicy, DensePolicy

def policy_grad(config, iterations = 150, num_processes = 8, num_trajs_per_process = 1, gamma = 0.99, **policy_params):
    env = make_multiple_env(**config)

    # determine the type of model
    if policy_params['model'] == 'dense':
        actor = DensePolicy(**policy_params['model_params'])
    elif policy_params['model'] == 'rnn':
        actor = RNNPolicy(**policy_params['model_params'])
    elif policy_params['model'] == 'attention':
        actor = AttentionPolicy(**policy_params['model_params'])
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
    config = configs.starter_config
    policy_params = params.dense_params

    policy_grad(config, iterations=150, num_processes=8, num_trajs_per_process=1, **policy_params)


if __name__ == '__main__':
    main()
