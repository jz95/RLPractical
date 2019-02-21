from ..base import BaseEnvironment


class MultiArmedBandit(BaseEnvironment):
    """ Multi-Armed Bandit Environment.
    A demo fot the 10-armed testbed case in page-28.
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
        BaseEnvironment.__init__(self, seed)
        self.n_arms = k
        self.reward_dist_means = means
        self.reward_dist_stds = stds

    def reward(self, action):
        """Bandit gives reward based on agent's action
        """
        mu = self.reward_dist_means[action]
        sig = self.reward_dist_stds[action]
        return self.random_state.normal(mu, sig)

    def _transit(self):
        pass

    def __repr__(self):
        name = '%d-Armed Bandit' % self.n_arms
        params = ['\tarm %d  Gaussian(%.2f, %.2f)' % (a, mu, sig)
                  for a, (mu, sig) in self.gaussian_params.items()]
        return name + '\n' + '\n'.join(params)


class NonStationaryMultiArmedBandit(MultiArmedBandit):
    """ NonStationary MultiArmedBadit.
    A demo for exercise-2.5 in page-33.
    The mean of each reward distribution would get an increment sampled
    from a zero-mean 0.01-std gaussian.
    """

    def __init__(self, k, means, stds, seed=None):
        MultiArmedBandit.__init__(self, k, means, stds, seed)

    def _update(self):
        noise = self.random_state.randn(self.n_arms) * 0.01
        self.reward_dist_means += noise

    def reward(self, action):
        """Bandit gives reward based on agent's action
        """
        self._update()
        return MultiArmedBandit.reward(self, action, None)
