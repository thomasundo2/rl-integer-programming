"""
similar to the actor classes, so that we can use the rollout_gen code
"""


import numpy as np
import torch


from helper import build_actor, build_critic
class PPO: # TODO: PPO is usually used with tanh activation functions, but they don't work with parallelization?
    def __init__(self, policy_params, critic_params, epochs, lr):

        self.policy = build_actor(policy_params)
        self.critic = build_critic(critic_params)

        if critic_params["model"] != "None":
            self.optimizer = torch.optim.Adam([
                {'params': self.policy.model.parameters(), 'lr': lr},
                {'params': self.critic.model.parameters(), 'lr': lr}
            ])
        else:
            self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=lr)


        self.policy_old = build_actor(policy_params)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.epochs = epochs
        self.MseLoss = torch.nn.MSELoss()

    def _compute_prob_torch(self, state):
        return self.policy_old._compute_prob_torch(state)

    def compute_prob(self, state):
        return self.policy_old.compute_prob(state)

    def _compute_loss(self, memory):
        # computes the loss of a single epoch

        for i, state in enumerate(memory.states):
            action = int(memory.actions[i])
            prob = self.policy._compute_prob_torch(state)
            prob_old =  self.policy_old._compute_prob_torch(state)

            k = len(prob)
            action_onehot = np.zeros(k)
            action_onehot[action] = 1
            action_onehot = torch.FloatTensor(action_onehot)

            prob_selected = torch.matmul(prob, action_onehot)
            prob_selected_old = torch.matmul(prob, action_onehot)





    def train(self, memory):
        # from here, memory.advantages must be computed, however rewards are preprocessed
        # todo: preprocessing needs to be normalized across the code
        for _ in range(self.epochs):









env_config = env_configs.hard_config
policy_params = gen_actor_params.gen_dense_params(m=60, n=60, t=50, lr=0.001)
critic_params = gen_critic_params.gen_no_critic()
