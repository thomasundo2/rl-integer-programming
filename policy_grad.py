from tqdm import tqdm
import numpy as np

# import wandb
# wandb.login()
# run=wandb.init(project="finalproject", entity="ieor-4575", tags=["n=10,m=20,i=1"])


from config import gen_actor_params, gen_critic_params, env_configs
from gymenv_v2 import make_multiple_env
from rollout_gen import RolloutGenerator
from pg_actors import AttentionPolicy, RNNPolicy, DensePolicy, RandomPolicy, DoubleAttentionPolicy
from pg_critics import DenseCritic, NoCritic
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

def build_critic(critic_params):
    if critic_params['model'] == 'dense':
        critic = DenseCritic(**critic_params['model_params'])
    elif critic_params['model'] == 'None':
        critic = NoCritic()
    else:
        raise NotImplementedError
    return critic



def policy_grad(env_config,
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
    logger = RewardLogger(env_config, policy_params, critic_params, hyperparams)
    rrecord = []

    env = make_multiple_env(**env_config)
    actor = build_actor(policy_params)
    critic = build_critic(critic_params)

    rollout_gen = RolloutGenerator(num_processes, num_trajs_per_process, verbose=False)



    for ite in tqdm(range(iterations)):
        memory = rollout_gen.generate_trajs(env, actor, gamma)

        critic.train(memory)
        baselines = critic.compute_values(memory.states)
        memory.advantages = np.array(memory.values) - baselines
        memory.advantages = (memory.advantages - np.mean(memory.advantages)) \
                            / (np.std(memory.advantages) + 1e-8)

        actor.train(memory)
        # log results
        rrecord.extend(memory.reward_sums)
        logger.record(memory.reward_sums) # writes record to a file

    plot_arr(rrecord, label="Moving Avg Reward " + policy_params['model'], window_size=101)
    plot_arr(rrecord, label="Reward " + policy_params['model'], window_size=1)


def main():
    env_config = env_configs.veryeasy_config
    policy_params = gen_actor_params.gen_attention_params(n=25, h=60, lr=0.001)
    critic_params = gen_critic_params.gen_critic_dense(m=25, n=25, t=20, lr=0.001)
    # critic_params = gen_critic_params.gen_critic_dense(m=20, n=10, t=10, lr=0.001)


    hyperparams = {"iterations": 750,  # number of iterations to run policy gradient
                   "num_processes": 12,  # number of processes running in parallel
                   "num_trajs_per_process": 1,  # number of trajectories per process
                   "gamma": 0.99  # discount factor
                   }
    print("DENSE CRITIC, ATTENTION POLICY")
    policy_grad(env_config,  # environment configuration
                policy_params,  # actor definition
                critic_params,
                **hyperparams
                )


    policy_params = gen_actor_params.gen_dense_params(m=25, n=25, t=20, lr=0.001)
    print("DENSE CRITIC, DENSE POLICY")
    policy_grad(env_config,  # environment configuration
                policy_params,  # actor definition
                critic_params,
                **hyperparams
                )



    #### try with more trajectories no critic

    critic_params = gen_critic_params.gen_no_critic()
    hyperparams = {"iterations": 350,  # number of iterations to run policy gradient
                   "num_processes": 12,  # number of processes running in parallel
                   "num_trajs_per_process": 2,  # number of trajectories per process
                   "gamma": 0.99  # discount factor
                   }

    policy_params = gen_actor_params.gen_dense_params(m=25, n=25, t=20, lr=0.001)

    policy_grad(env_config,  # environment configuration
                policy_params,  # actor definition
                critic_params,
                **hyperparams
                )

    policy_params = gen_actor_params.gen_attention_params(n=25, h=60, lr=0.001)

    policy_grad(env_config,  # environment configuration
                policy_params,  # actor definition
                critic_params,
                **hyperparams
                )

    critic_params = gen_critic_params.gen_critic_dense(m=25, n=25, t=20, lr=0.001)
    policy_params = gen_actor_params.gen_dense_params(m=25, n=25, t=20, lr=0.001)

    policy_grad(env_config,  # environment configuration
                policy_params,  # actor definition
                critic_params,
                **hyperparams
                )

    policy_params = gen_actor_params.gen_attention_params(n=25, h=60, lr=0.001)

    policy_grad(env_config,  # environment configuration
                policy_params,  # actor definition
                critic_params,
                **hyperparams
                )










if __name__ == '__main__':
    main()
