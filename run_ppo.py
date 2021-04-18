from tqdm import tqdm
import numpy as np

# import wandb
# wandb.login()
# run=wandb.init(project="finalproject", entity="ieor-4575", tags=["n=10,m=20,i=1"])


from config import gen_actor_params, gen_critic_params, env_configs
from gymenv_v2 import make_multiple_env
from rollout_gen import RolloutGenerator
from helper import plot_arr
from build_ppo import build_ppo
from logger import RewardLogger

def run_ppo(env_config,
            policy_params,
            critic_params,
            iterations=150,
            num_processes=8,
            num_trajs_per_process=1,
            gamma=0.99
            ):
    hyperparams = {"iterations": iterations,
                   "num_processes": num_processes,
                   "num_trajs_per_process": num_trajs_per_process,
                   "gamma": gamma
                   }
    logger = RewardLogger(env_config, policy_params, critic_params, hyperparams, ppo_tag=True)
    rrecord = []

    env = make_multiple_env(**env_config)
    ppo_ac = build_ppo(policy_params, critic_params)

    rollout_gen = RolloutGenerator(num_processes, num_trajs_per_process, verbose=False)

    for _ in tqdm(range(iterations)):
        memory = rollout_gen.generate_trajs(env, ppo_ac, gamma)

        ppo_ac.train(memory)
        # log results
        rrecord.extend(memory.reward_sums)
        logger.record(memory.reward_sums)

    plot_arr(rrecord, label="Moving Avg Reward " + policy_params['model'], window_size=101)
    plot_arr(rrecord, label="Reward " + policy_params['model'], window_size=1)


def main():
    env_config = env_configs.starter_config
    policy_params = gen_actor_params.gen_dense_params(m=20, n=10, t=10, lr=0.001)
    critic_params = gen_critic_params.gen_no_critic()


    hyperparams = {"iterations": 500,  # number of iterations to run policy gradient
                   "num_processes": 12,  # number of processes running in parallel
                   "num_trajs_per_process": 1,  # number of trajectories per process
                   "gamma": 0.99  # discount factor
                   }
    run_ppo(env_config,  # environment configuration
                policy_params,  # actor definition
                critic_params,
                **hyperparams
                )







if __name__ == '__main__':
    main()
