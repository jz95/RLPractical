from abc import ABC, abstractmethod
from numpy.random import RandomState
from . import utilis
import numpy as np


class Agent(ABC):
    """ Abstract Class for Agent.
    """

    def __init__(self, seed):
        self.random_state = RandomState(seed)
        self.actions = []
        self.rewards = []

    @abstractmethod
    def update(self):
        """ Update agent's params.
        """
        raise NotImplemented

    @abstractmethod
    def action(self):
        """ Agent selects action.
        """
        raise NotImplemented

    def get_reward(self, Rt):
        """ Agent gets reward from Environment.
        """
        self.Rt = Rt
        self.rewards.append(Rt)

    @abstractmethod
    def _step_size(self):
        raise NotImplemented

    # @abstractmethod
    # def reset(self):
    #     raise NotImplemented


class ActionValueMethod(Agent):
    """ Abstract Class for Action Value Methods
    """

    def __init__(self, Q0, seed):
        Agent.__init__(self, seed)
        self.aciton_value_estimate = Q0
        self.n_arms = len(Q0)
        self.action_cnt = np.zeros((self.n_arms, ))

        self.timestep = 0

        self.At = None
        self.Rt = 0

    def update(self):
        Qt = self.aciton_value_estimate[self.At]
        newQ = Qt + self._step_size() * (self.Rt - Qt)
        self.aciton_value_estimate[self.At] = newQ
        self.timestep += 1


class EpsGreedy(ActionValueMethod):
    """ EpsGreedy Agent.
    """

    def __init__(self, eps, Q0, seed=None):
        """
        Params:
        eps - prob. of exploration (being totally greedy when eps = 0).
        Q0 - initial estimate for action value. Can be a list or numpy array.
        seed - random seed.
        """
        ActionValueMethod.__init__(self, Q0, seed)
        self.eps = eps

    def action(self):
        # exploit
        if self.random_state.uniform(0, 1) >= self.eps:
            At = np.argmax(self.aciton_value_estimate)
        # explore
        else:
            At = self.random_state.choice(range(self.n_arms))
        self.At = At
        self.actions.append(At)
        self.action_cnt[At] += 1
        return At

    def _step_size(self):
        return 1 / self.action_cnt[self.At]

    def __repr__(self):
        if self.eps == 0:
            return 'Greedy Agent Using Sample Average'
        else:
            return 'Eps Greedy Agent Using Sample Average(eps = %.2f)' % self.eps


class EpsGreedyConstStep(EpsGreedy):
    """ EpsGreedy Agent Using Constant Step Size.
    """

    def __init__(self, eps, Q0, alpha, seed=None):
        """Params:
        eps - prob. of exploration (being totally greedy when eps = 0).
        Q0 - initial estimate for action value. Can be a list or numpy array.
        alpha - const step size.
        seed - random seed.
        """
        assert alpha <= 1 and alpha > 0
        EpsGreedy.__init__(self, eps, Q0, seed)
        self.alpha = alpha

    # step size is a constatnt alpha
    def _step_size(self):
        return self.alpha

    def __repr__(self):
        if self.eps == 0:
            return 'Greedy Agent with Const Step(alpha = %.2f)' % self.alpha
        else:
            return 'Eps Greedy Agent with Const Step(eps = %.2f, alpha = %.2f)' \
                % (self.eps, self.alpha)


class UCB(ActionValueMethod):
    """Upper Confidence Bound (UCB) Action Selection.
    """

    def __init__(self, alpha, conf_level, Q0, seed=None):
        """Params:
        alpha - const step size.
        conf_level - confident level.
        Q0 - initial estimate for action value. Can be a list or numpy array.
        seed - random seed.
        """
        ActionValueMethod.__init__(self, Q0, seed)
        self.c = conf_level
        self.alpha = alpha

    def action(self):
        if 0 in self.action_cnt:
            zeroA = np.argwhere(self.action_cnt == 0).ravel()
            At = np.random.choice(zeroA)
        else:
            t_arr = self.timestep * np.ones((self.n_arms, ))
            tmp = self.aciton_value_estimate + self.c * \
                np.sqrt(np.log(t_arr) / self.action_cnt)
            # after exploring all zero-cnt actions
            At = np.argmax(tmp)

        self.At = At
        self.actions.append(At)
        self.action_cnt[At] += 1
        return At

    def _step_size(self):
        return self.alpha

    def __repr__(self):
        return 'UCB(c = %.2f, alpha=%.2f)' % (self.c, self.alpha)


class GradientBandit(Agent):
    """Gradient Bandit Algorithm.
    """

    def __init__(self, H0, alpha, baseline=False, seed=None):
        """Params:
        H0 - initial estimate for action value. Can be a list or numpy array.
        alpha - const step size.
        baseline - use baseline or not.
        seed - random seed.
        """
        Agent.__init__(self, seed)
        self.Hs = H0
        self.prob = utilis.softmax(self.Hs)
        self.n_arms = len(H0)
        self.use_baseline = baseline
        self.alpha = alpha

        self.timestep = 0

        self.At = None
        self.Rt = 0

    def update(self):
        baseline = np.mean(self.rewards) if self.use_baseline else 0
        self.Hs += self._step_size() * (self.Rt - baseline) * \
            ((np.arange(self.n_arms) == self.At) - self.prob)

        self.prob = utilis.softmax(self.Hs)
        self.timestep += 1

    def action(self):
        # ramdom sample from the softmax distribution
        At = self.random_state.choice(range(self.n_arms), p=self.prob)
        self.At = At
        self.actions.append(At)
        return At

    def _step_size(self):
        return self.alpha

    def __repr__(self):
        return 'GradientBandit(alpha = %.2f)' % self.alpha
