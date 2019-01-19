from abc import abstractmethod
from collections import Counter
import math
from . import utilis
from .rlpBase import Agent, Environment


class MultiArmedBandit(Environment):
    def __init__(self, k, means, stds, seed=None):
        Environment.__init__(self, seed)
        self.n_arms = k
        self.gaussian_params = dict([
            (a,
                (mu, sig)
             ) for a, (mu, sig) in
            enumerate(zip(means, stds))
        ])

    def reward(self, action):
        mu, sig = self.gaussian_params[action]
        return self.random_state.normal(mu, sig)


class ActionValueMethod(Agent):
    """ Abstract Class for Action Value Methods
    """

    def __init__(self, seed, Q0):
        Agent.__init__(self, seed)
        self.aciton_value_estimate = Q0
        self.action_cnt = Counter()
        self.action_set = set(Q0.keys())

        self.timestep = 0

        self.At = None
        self.Rt = 0

    def update(self):
        Qt = self.aciton_value_estimate[self.At]
        newQ = Qt + self._step_size() * (self.Rt - Qt)
        self.aciton_value_estimate[self.At] = newQ
        self.timestep += 1

    @abstractmethod
    def _step_size(self):
        raise NotImplemented


class EpsGreedy(ActionValueMethod):
    def __init__(self, eps, seed):
        ActionValueMethod.__init__(self, seed)
        self.eps = 0

    def action(self):
        if self.random_state.uniform(0, 1) < self.eps:
            At = self._exploit()
        else:
            At = self._explore()
        self.At = At
        self.actions.append(At)
        return At

    def get_reward(self, Rt):
        self.Rt = Rt
        self.rewards.append(Rt)

    def _exploit(self):
        return utilis.argmax(self.aciton_value_estimate)

    def _explore(self):
        return self.random_state.choice(self.action_set)

    def _step_size(self):
        return 1 / self.action_cnt[self.At]


class NonStationaryEpsGreedy(EpsGreedy):
    def __init__(self, eps, seed, alpha):
        assert alpha <= 1 and alpha > 0
        EpsGreedy.__init__(self, eps, seed)
        self.alpha = alpha

    def _step_size(self):
        return self.alpha


class UCB(ActionValueMethod):
    def __init__(self, conf_level, seed):
        ActionValueMethod.__init__(self, seed)
        self.c = conf_level

    def action(self):
        tmp = {}
        for At, cnt in self.action_cnt:
            if cnt == 0:
                self.At = At
                self.actions.append(At)
                return At
            else:
                Qt = self.aciton_value_estimate[At]
                tmp[At] = Qt +\
                    self.c * math.sqrt(math.log(self.timestep) / cnt)
        # after exploring all zero-cnt actions
        At = utilis.argmax(tmp)
        self.At = At
        self.actions.append(At)
        return At

    def _step_size(self):
        return 1 / self.action_cnt[self.At]


class GradientBandit(Agent):
    pass


class Experiment:
    def __init__(self):
        pass







