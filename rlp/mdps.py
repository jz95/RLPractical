from .base import BaseEnvironment
from abc import abstractmethod


class MDP(BaseEnvironment):
    """ Abstract Environment Class for Discrete Finite Markov Decision Processes.
    """

    def __init__(self, seed):
        super(MDP, self).__init__(seed)

    @abstractmethod
    def prob_next_state_n_reward(self, state, action):
        """ get \\sum_{s', r} p(s', r| s, a) [r + \\gamma V(s')]
        Params:
        state - given state
        action - given action

        Returns:
        joint probability distribution over next_state and reward
        dict: (next_state, reward) => prob, type: tuple => float
        """
        raise NotImplementedError
