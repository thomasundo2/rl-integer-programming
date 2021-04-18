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

        # todo: create parameters for these hardcoded values
        self.eps_clip = 0.2
        self.entropy_coeff = 0.01

    def _compute_prob_torch(self, state):
        return self.policy_old._compute_prob_torch(state)

    def compute_prob(self, state):
        return self.policy_old.compute_prob(state)

    def compute_val(self, state):
    def _compute_loss(self, memory):
        # computes the loss of a single epoch, given a set of trajectories
        loss = 0
        for i, state in enumerate(memory.states):
            # compute advantage
            baseline = self.critic._compute_value_torch(state)
            advantage = memory.values[i] - baseline.detach()

            action = int(memory.actions[i])
            prob = self.policy._compute_prob_torch(state)
            prob_old =  self.policy_old._compute_prob_torch(state).detach()

            dist_entropy = torch.distributions.Categorical(prob).entropy()

            k = len(prob)
            action_onehot = np.zeros(k)
            action_onehot[action] = 1
            action_onehot = torch.FloatTensor(action_onehot)

            prob_selected = torch.matmul(prob, action_onehot)
            prob_selected_old = torch.matmul(prob_old, action_onehot)

            ratio = prob_selected / prob_selected_old

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss += -torch.min(surr1, surr2) + 0.5 * self.MseLoss(baseline, memory.values[i]) - 0.01 * dist_entropy
        loss = -loss / len(memory.states)
        assert loss.requires_grad == True
        return loss


    def train(self, memory):
        # from here, memory.advantages must be computed, however rewards are preprocessed
        # todo: preprocessing needs to be normalized across the code
        for _ in range(self.epochs):










env_config = env_configs.hard_config
policy_params = gen_actor_params.gen_dense_params(m=60, n=60, t=50, lr=0.001)
critic_params = gen_critic_params.gen_no_critic()
