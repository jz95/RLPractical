from .. import utilis
from ..base import BaseAgent
from abc import abstractmethod
import numpy as np


class MultiArmedBanditBaseAgent(BaseAgent):
    """ Abstract Class for Multi Armed Bandit Agent
    """

    def __init__(self, seed):
        BaseAgent.__init__(seed)
        self.rewards = []
        self.actions = []

    def get_reward(self, Rt):
        """ A simple method to get a scalar reward from the environment.
        """
        self.get_response(Rt, S=None)

    def get_response(self, R, S):
        """ Get response from env, including reward and new state.
        Please use get_reward().
        ==========================
        Params:
        R - scalar reward.
        S - new state, which is ignored in this case.
        """
        self.rewards.append(R)

    @abstractmethod
    def _step_size(self):
        return self.alpha


class ActionValueMethod(MultiArmedBanditBaseAgent):
    """ Abstract Class for Action Value Methods
    """

    def __init__(self, Q0, seed):
        MultiArmedBanditBaseAgent.__init__(self, seed)
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


class GradientBandit(MultiArmedBanditBaseAgent):
    """Gradient Bandit Algorithm.
    """

    def __init__(self, H0, alpha, baseline=False, seed=None):
        """Params:
        H0 - initial estimate for action value. Can be a list or numpy array.
        alpha - const step size.
        baseline - use baseline or not.
        seed - random seed.
        """
        MultiArmedBanditBaseAgent.__init__(self, seed)
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
