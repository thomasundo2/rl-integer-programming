class Memory(object):
    """general memory class that can hold states, actions, rewards, and values
    """

    def __init__(self):
        self.states = []  # each element of [Ab, c0, cuts]
        self.actions = []
        self.rewards = []  # each element contains trajectory of raw rewards
        self.isdone = []
        self.values = []  # discounted reward

    def clear(self):
        self.__init__()


class TrajMemory(Memory):
    def add_frame(self, condensed_s, a, r):
        self.states.append(condensed_s)
        # self.isdone.append(d)
        self.actions.append(a)
        self.rewards.append(r)


class MasterMemory(Memory):
    """inherits a general memory class, through which we can easily add trajectories to the MasterMemory
    """

    def add_trajectory(self, trajectory_memory):
        self.states.extend(trajectory_memory.states)
        self.actions.extend(trajectory_memory.actions)
        # self.isdone.extend(trajectory_memory.isdone)
        self.rewards.extend(trajectory_memory.rewards)
        self.values.extend(trajectory_memory.values)
