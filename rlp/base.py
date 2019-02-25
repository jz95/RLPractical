from abc import ABC, abstractmethod
from numpy.random import RandomState


class BaseAgent(ABC):
    """ Abstract Class for Agent.
    """

    def __init__(self, seed):
        self.random_state = RandomState(seed)

    @abstractmethod
    def update(self):
        """ Update agent's params.
        """
        raise NotImplemented

    @abstractmethod
    def action(self):
        """ Agent selects action based on its policy.
        """
        raise NotImplemented

    @abstractmethod
    def set_experience(self, reward, new_state):
        """ Agent gets reward and new state from Environment.
        """
        NotImplemented


class BaseEnvironment(ABC):
    """ Abstract Class for Environment.
    """

    def __init__(self, seed):
        self.random_state = RandomState(seed)

    @abstractmethod
    def react(self, state, action):
        """ Give a scalar reward and next state based on current action and state.
        """
        raise NotImplemented
