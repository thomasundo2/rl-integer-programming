import torch
import numpy as np

from basepolicy import AbstractPolicy

class AttentionPolicy(AbstractPolicy):
    def __init__(self, n, h, lr):
        """
        n: size of constraints and the b
        h: size of output
        """
        self.model = torch.nn.Sequential(
            # input layer
            torch.nn.Linear(n+1, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, h)
        )

        # DEFINE THE OPTIMIZER
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # RECORD HYPER-PARAMS
        self.n = n
        self.h = h

    def _compute_prob_torch(self, state):
        Ab, c0, cuts = state
        Ab_h = self.model(torch.FloatTensor(np.array(Ab, dtype=np.float)))
        cuts_h = self.model(torch.FloatTensor(np.array(cuts, dtype=np.float)))

        scores = torch.mean(torch.matmul(Ab_h, torch.transpose(cuts_h, 0, 1)), 0)

        scores = (scores - np.mean()) / (np.std(scores) + 1e-8)
        prob = torch.nn.functional.softmax(scores, dim=0)
        return prob

class RNNPolicy(AbstractPolicy):
    def __init__(self, num_dec_vars, lr):
        """
        num_dec_vars is n
        """
        self.rnn_model = self.RecurrentNetwork(num_dec_vars+1)
        self.optimizer = torch.optim.Adam(self.rnn_model.parameters(), lr=lr)

    def _compute_prob_torch(self, state):
        Ab, _, cuts = state
        Ab_numpy = np.array(Ab)
        batch = []
        for cut in cuts:
            x = np.append(Ab.flatten(), cut.flatten())
            batch.append(x)

        batch_torch = torch.FloatTensor(batch)
        scores = self.rnn_model(batch_torch).flatten()

        return torch.nn.functional.softmax(scores, dim=-1)

    class RecurrentNetwork(torch.nn.Module):
        def __init__(self, num_dec_vars):
            super(self.RecurrentNetwork, self).__init__()
            self.rnn = torch.nn.GRU(num_dec_vars, 64, num_layers=2,
                                    batch_first=True, dropout=0.1)
            self.ffnn = torch.nn.Linear(64, 1)

        def forward(self, states):  # takes in batch of states of variable length
            # lens = (x != 0).sum(1)
            # p_embeds = rnn.pack_padded_sequence(embeds, lens, batch_first=True, enforce_sorted=False)
            _, hn = self.rnn(states)
            hns = hn.split(1, dim=0)
            last_hn = hns[-1]
            scores = self.ffnn(last_hn.squeeze(0))
            prob = torch.nn.functional.softmax(scores, dim=0)
            return prob

class DensePolicy(AbstractPolicy):
    def __init__(self, m, n, t, lr):
        """
        max_input is the size of the maximum state + size of maximum action
        Let t be the max number of timesteps, ie the max number of cuts added
        maximum state/action size: (m + t - 1 + 1, n+1)
        """
        self.model = torch.nn.Sequential(
            # input layer
            torch.nn.Linear((m + t) * (n + 1), 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.m = m
        self.t = t
        self.n = n
        self.full_length = (m + t) * (n + 1)

        # DEFINE THE OPTIMIZER
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _compute_prob_torch(self, state):
        Ab, c0, cuts = state
        scores = torch.FloatTensor()
        batch = []
        for cut in cuts:
            x = np.append(Ab.flatten(), cut.flatten())
            padded_x = np.append(x, np.zeros(self.full_length - len(x))) # pad with zeros

            batch.append(padded_x)

        batch_torch = torch.FloatTensor(batch)
        scores = self.model(batch_torch).flatten()

        return torch.nn.functional.softmax(scores, dim=-1)

