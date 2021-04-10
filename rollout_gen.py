import numpy as np
import os
import sys
from contextlib import contextmanager
from multiprocessing import Pool


from memory import TrajMemory, MasterMemory
from helper import discounted_rewards
class RolloutGenerator(object):
    """a parallel process trajectory generator
    """

    def __init__(self, num_processes, num_trajs_per_process, verbose = False):
        self.num_processes = num_processes
        self.num_trajs_per_process = num_trajs_per_process
        self.verbose = verbose # prints the sum of the rewards for each trajectory

    def _condense_state(self, s):
        """Takes A, b, c0, cuts_a, cuts_b and concatenates Ab and cuts
        """

        def append_col(A, b):
            expanded_b = np.expand_dims(b, 1)
            return np.append(A, expanded_b, 1)

        A, b, c0, cuts_a, cuts_b = s
        Ab = append_col(A, b)
        cuts = append_col(cuts_a, cuts_b)
        return (Ab, c0, cuts)

    def _generate_traj_process(self, env, actor, gamma, process_num):
        @contextmanager # supress Gurobi message
        def suppress_stdout():
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout

        np.random.seed()
        trajs = []
        for num in range(self.num_trajs_per_process):
            with suppress_stdout():  # remove the Gurobi std out
                s = env.reset()  # samples a random instance every time env.reset() is called
            condensed_s = self._condense_state(s)
            d = False
            t = 0
            traj_memory = TrajMemory()
            rews = 0
            while not d:
                actsize = len(condensed_s[-1])  # k
                prob = actor.compute_prob(condensed_s)
                prob /= np.sum(prob)

                a = np.random.choice(actsize, p=prob.flatten())

                new_s, r, d, _ = env.step([a])
                rews += r
                t += 1

                r = r * 10
                traj_memory.add_frame(condensed_s, a, r)

                if not d:
                    condensed_s = self._condense_state(new_s)
            if self.verbose:
                print(f"[{process_num}] rews: {rews} \t t: {t}")
            Q = discounted_rewards(traj_memory.rewards, gamma)
            val = np.array(Q)
            traj_memory.values = val

            trajs.append(traj_memory)
        return trajs

    def generate_trajs(self, env, actor, gamma):
        if self.num_processes == 1: # don't run in parallel
            DATA = []
            DATA.append(self._generate_traj_process(env, actor, gamma, 0))
        else:
            env_list = [env] * self.num_processes
            actor_list = [actor] * self.num_processes
            gamma_list = [gamma] * self.num_processes
            i_list = np.arange(self.num_processes)
            with Pool(processes=self.num_processes) as pool:
                DATA = pool.starmap(self._generate_traj_process,
                                    zip(env_list,
                                        actor_list,
                                        gamma_list,
                                        i_list
                                        )
                                    )
            # unpack data
        master_mem = MasterMemory()
        for trajs in DATA:
            for traj in trajs:
                master_mem.add_trajectory(traj)
        return master_mem
