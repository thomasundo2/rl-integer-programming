from tqdm import tqdm
import numpy as np

# import wandb
# wandb.login()
# run=wandb.init(project="finalproject", entity="ieor-4575", tags=["n=10,m=20,i=1"])


from config import gen_actor_params, gen_critic_params, gen_rnd_params, env_configs
from gymenv_v2 import make_multiple_env
from rollout_gen import RolloutGenerator
from helper import plot_arr
from build_ppo import build_ppo
from build_rnd import build_rnd
from logger import RewardLogger

def run_ppo(env_config,
            policy_params,
            critic_params,
            rnd_params,
            iterations=150,
            num_processes=8,
            num_trajs_per_process=1,
            gamma=0.99,
            intrinsic_gamma = 0.999
            ):
    hyperparams = {"iterations": iterations,
                   "num_processes": num_processes,
                   "num_trajs_per_process": num_trajs_per_process,
                   "gamma": gamma
                   }
    logger = RewardLogger(env_config, policy_params, critic_params, rnd_params, hyperparams, ppo_tag=True)
    rrecord = []

    env = make_multiple_env(**env_config)
    ppo_ac = build_ppo(policy_params, critic_params)
    rnd = build_rnd(rnd_params)

    rollout_gen = RolloutGenerator(num_processes, num_trajs_per_process, verbose=True)
    for ite in tqdm(range(iterations)):
        memory = rollout_gen.generate_trajs(env, ppo_ac, rnd, gamma, intrinsic_gamma)
        memory.values = (memory.values - np.mean(memory.values)) \
                            / (np.std(memory.values) + 1e-8)
        memory.intrinsic_values = (memory.intrinsic_values - np.mean(memory.intrinsic_values)) \
                            / (np.std(memory.intrinsic_values) + 1e-8)
        rnd.train(memory)
        ppo_ac.train(memory)
        # log results
        rrecord.extend(memory.reward_sums)
        logger.record(memory.reward_sums)

        if ite > 0 and ite % 100 == 0:
            logger.save_rnd(rnd, ite)
            logger.save_ppo(ppo_ac, ite)


    plot_arr(rrecord, label="Moving Avg Reward " + policy_params['model'], window_size=101)
    plot_arr(rrecord, label="Reward " + policy_params['model'], window_size=1)


def main():
    env_config = env_configs.easy_config
    #policy_params = gen_actor_params.gen_dense_params(m=20, n=10, t=10, lr=0.001)
    policy_params = gen_actor_params.gen_dense_params(m=60, n=60, t=50, lr=0.001)
    #critic_params = gen_critic_params.gen_critic_dense(m=15, n=15, t=20, lr=0.001)
    critic_params = gen_critic_params.gen_no_critic()
    rnd_params = gen_rnd_params.gen_no_rnd()

    hyperparams = {"iterations": 10000,  # number of iterations to run policy gradient
                   "num_processes": 6,  # number of processes running in parallel
                   "num_trajs_per_process": 1,  # number of trajectories per process
                   "gamma": 0.99,  # discount factor
                   "intrinsic_gamma": 0.999 # intrinsic reward discount factor
                   }
    run_ppo(env_config,  # environment configuration
                policy_params,  # actor definition
                critic_params,
                rnd_params,
                **hyperparams
                )

if __name__ == '__main__':
    main()
