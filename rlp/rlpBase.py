from abc import ABC, abstractmethod
from numpy.random import RandomState


class Agent(ABC):
    def __init__(self, seed):
        self.random_state = RandomState(seed)
        self.actions = []
        self.rewards = []

    @abstractmethod
    def update(self):
        """ abstract method for updating params.
        """
        raise NotImplemented

    @abstractmethod
    def action(self):
        """ abstract method for action.
        """
        raise NotImplemented

    @abstractmethod
    def reset(self):
        raise NotImplemented


class Environment(ABC):
    def __init__(self, seed):
        self.random_state = RandomState(seed)

    @abstractmethod
    def reward(self, action):
        """ give reward based on agent action.
        should return a scalar.
        """
        raise NotImplemented


class Simulation(ABC):
    def __init__(self, env, agent, seed):
        self.env = env
        self.agent = agent

    def _reset(self):
        pass

    def run(self):
        pass

    def report(self):
        pass
