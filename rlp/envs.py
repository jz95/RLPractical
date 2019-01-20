from abc import ABC, abstractmethod
from numpy.random import RandomState


class Environment(ABC):
    """ Abstract Class for Environment.
    """

    def __init__(self, seed):
        self.random_state = RandomState(seed)

    @abstractmethod
    def reward(self, action):
        """ Give a scalar reward based on agent action.
        """
        raise NotImplemented


class MultiArmedBandit(Environment):
    """ Multi-Armed Bandit Environment.
    Give rewards sampled from Gaussian distribution.
    """

    def __init__(self, k, means, stds, seed=None):
        """
        Params:
        ========================
        k - number of arms, i.e. number of gaussian distributions.
        means - list of mean values of k gaussian distributions.
        stds - list of standard deviations of k gaussian distributions.
        seed - random seed.
        """
        Environment.__init__(self, seed)
        self.n_arms = k
        self.reward_dist_means = means
        self.reward_dist_stds = stds

    def reward(self, action):
        mu = self.reward_dist_means[action]
        sig = self.reward_dist_stds[action]
        return self.random_state.normal(mu, sig)

    def __repr__(self):
        name = '%d-Armed Bandit' % self.n_arms
        params = ['\tarm %d  Gaussian(%.2f, %.2f)' % (a, mu, sig)
                  for a, (mu, sig) in self.gaussian_params.items()]
        return name + '\n' + '\n'.join(params)


class NonStationaryMultiArmedBandit(MultiArmedBandit):
    """ NonStationary MultiArmedBadit.
    The mean of each reward distribution would get an increment sampled
    from a zero-mean 0.01-std gaussian.
    """

    def __init__(self, k, means, stds, seed=None):
        MultiArmedBandit.__init__(self, k, means, stds, seed)

    def _update(self):
        noise = self.random_state.randn(self.n_arms) * 0.01
        self.reward_dist_means += noise

    def reward(self, action):
        self._update()
        return MultiArmedBandit.reward(self, action)
