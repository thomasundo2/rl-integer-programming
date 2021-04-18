from ppo import PPOPolicy

def build_ppo(policy_params, critic_params):
    epochs = 10
    lr = 0.01
    eps_clip = 0.2
    entropy_coeff = 0.01

    return PPOPolicy(policy_params, critic_params, epochs, lr, eps_clip, entropy_coeff)
