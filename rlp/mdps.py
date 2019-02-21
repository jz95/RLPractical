from .base import BaseAgent, BaseEnvironment
import numpy as np


class MDP(BaseEnvironment):
    """ Abstract Environment Class for Discrete Finite Markov Decision Processes.
    """

    def __init__(self, action, state, state_transition_prob, seed):
        BaseEnvironment.__init__(self, seed)
        self.action_set = set(action)
        self.state_set = set(state)

        self.transition_prob = state_transition_prob

    def _transit(self):
        key = (self.state, self.action)
        new_state = self.random_state.choice(p=self.transition_prob[key])
        reward = 
        return new_state, reward

    def probNextStates(self, initState, action):
        pass


class MDP(MDP):
    """ Abstract Environment Class for Finite Markov Decision Processes.
    """

    def __init__(self, n_action, n_state, state_transition_prob, seed):
        BaseEnvironment.__init__(self, seed)
        self.action_set = np.arange(n_action)
        self.state_set = np.arange(n_state)

        self.transition_prob = state_transition_prob

    def _transit(self):
        key = (self.state, self.action)
        new_state = self.random_state.choice(p=self.transition_prob[key])



class MDPAgent(BaseAgent):
    """Abstract Class for Agents in finite Markov Decision Processes.
    """

    def __init__(self, n_state, n_action, seed):
        """
        Params:
        n_state - number of finite states.
        n_action - number of finite actions.
        seed - random seed.
        """
        BaseAgent.__init__(self, seed)

        self.n_action = n_action
        self.n_state = n_state

        # state value function
        self.V = np.zeros(n_state)
        # action-value function
        self.Q = np.zeros((n_state, n_action))
        self.policy = np.ones((n_state, n_action)) / n_action

    def action(self, action, state):
        At = self.random_state.choice(
            self.n_action, size=1, p=self.policy[state])
        self.actions.append(At)
        return At
