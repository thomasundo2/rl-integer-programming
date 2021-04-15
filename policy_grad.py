from tqdm import tqdm
import numpy as np

# import wandb
# wandb.login()
# run=wandb.init(project="finalproject", entity="ieor-4575", tags=["n=10,m=20,i=1"])


from config import configs, params
from gymenv_v2 import make_multiple_env
from rollout_gen import RolloutGenerator
from policies import AttentionPolicy, RNNPolicy, DensePolicy, RandomPolicy, DoubleAttentionPolicy
from helper import plot_arr
from logger import RewardLogger


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
    elif policy_params['model'] == 'double_attention':
        actor = DoubleAttentionPolicy(**policy_params['model_params'])
    else:
        raise NotImplementedError

    return actor


def policy_grad(env_config,
                policy_params,
                iterations=150,
                num_processes=8,
                num_trajs_per_process=1,
                gamma=0.99
                ):
    env = make_multiple_env(**env_config)

    actor = build_actor(policy_params)

    rollout_gen = RolloutGenerator(num_processes, num_trajs_per_process, verbose=True)


    # todo: temporary until hyperparameters are reformatted
    hyperparams = {"iterations": iterations,
                   "num_processes": num_processes,
                   "num_trajs_per_process": num_trajs_per_process,
                   "gamma": gamma
                   }

    logger = RewardLogger(env_config, policy_params, hyperparams)
    baseline = 0

    rrecord = []

    for ite in tqdm(range(iterations)):
        if ite % 10 == 0 and ite > 0:
            print(np.mean(rrecord[-10:]))
        memory = rollout_gen.generate_trajs(env, actor, gamma, baseline)
        # baseline = np.mean(memory.rewards)
        memory.values = (memory.values - np.mean(memory.values)) / (np.std(memory.values) + 1e-8)

        loss = actor.train(memory)

        # log results
        rrecord.extend(memory.reward_sums)
        logger.record(memory.reward_sums) # writes record to a file

    plot_arr(rrecord, label="Moving Avg Reward " + policy_params['model'], window_size=101)
    plot_arr(rrecord, label="Reward " + policy_params['model'], window_size=1)


def main():
    env_config = configs.easy_config
    #policy_params = params.gen_rnn_params(n=10, lr = 0.001)
    # policy_params = params.gen_dense_params(m = 60, n=60, t = 50, lr=0.001)
    #
    # policy_params = params.gen_rand_params()

    # policy_params = params.gen_dense_params(m = 60, n=60, t = 50, lr=0.001)
    policy_params = params.gen_attention_params(n = 60, h = 32)

    hyperparams = {"iterations": 1000,  # number of iterations to run policy gradient
                   "num_processes": 12,  # number of processes running in parallel
                   "num_trajs_per_process": 1,  # number of trajectories per process
                   "gamma": 0.99  # discount factor
                   }

    policy_grad(env_config,  # environment configuration
                policy_params,  # actor definition
                **hyperparams
                )


if __name__ == '__main__':
    main()
