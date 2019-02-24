from rlp import utilis
from .base import JackCarRentalAgent, JackCarRentalEnv, DPGridWorldAgent, DPGridWorldEnv
import numpy as np


class DynamicProgrammingSolver:
    """dp solver.
    """

    def __init__(self, agent, model, threshold=1e-3):
        """
        Params:
        agent - DynamicProgrammingAgent object.
        model - DynamicProgrammingEnvModel object.
        threshold - threshold for loop termination.
        seed - random seed.
        """
        self.agent = agent
        self.model = model
        self.threshold = threshold

    def policy_eval(self, onestep=False):
        """ DP policy evaluation. see page 74.
        Params:
        onestep - run evalution until converge or just run onestep, if True, only run one-step
        """
        while True:
            diff = 0
            for state in self.agent.policy:
                oldV = self.agent.V[state]
                # \sum \pi(a|s) \sum_{s', r} p(s', r| s, a)[r + \gamma V(s')]
                newV = sum(utilis.element_wise_product(self._expectation_by_action(state), self.agent.policy[state]).values())
                self.agent.V[state] = newV
                diff = max(diff, abs(oldV - newV))
            if diff < self.threshold or onestep:
                break

    def _expectation_by_action(self, state):
        """ helper function to compute the expecation by action in DP update.
        i.e.
            \\pi(a|s) \\sum_{s', r} p(s', r| s, a)[r + \\gamma V(s')]
        """
        ret = {}
        for action in self.agent.policy[state]:
            # p(s', r| s, a)
            probs_on_nextState_reward = self.model.prob_next_state_n_reward(state, action)
            s = 0
            for (nextState, reward), p in probs_on_nextState_reward.items():
                s += (reward + self.agent.discountRatio * self.agent.V[nextState]) * p
            ret[action] = s
        return ret

    def policy_improve(self):
        """ DP policy improvement. see page 76.
        """
        for state in self.agent.policy:
            max_a = utilis.argmax(self._expectation_by_action(state))
            for action in self.agent.policy[state]:
                if action == max_a[0]:
                    self.agent.policy[state][action] = 1
                else:
                    self.agent.policy[state][action] = 0


class DPGridWorldSolver(DynamicProgrammingSolver):
    """Dynamic programming GridWorld Solver. see page 76.
    """

    def __init__(self, agent, model, threshold=1e-3):
        """
        Params:
        agent - Gridworld Agent object.
        model - Gridworld env object.
        threshold - threshold for loop termination.
        seed - random seed.
        """
        assert isinstance(agent, DPGridWorldAgent)
        assert isinstance(model, DPGridWorldEnv)
        super(DPGridWorldSolver, self).__init__(agent, model, threshold)


class JackCarRentalSolver(DynamicProgrammingSolver):
    """ DP sovler for Jack's Car Rental Problem in page-81.
    """

    def __init__(self, agent, model, threshold=1e-3):
        assert isinstance(agent, JackCarRentalAgent)
        assert isinstance(model, JackCarRentalEnv)
        super(JackCarRentalSolver, self).__init__(agent, model, threshold)
