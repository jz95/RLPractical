from ..grid_world import *
from .. import utilis


class DPGridWorldSolver(GridWorldAgent):
    """Dynamic programming GridWorld Agent. see page 76.
    """

    def __init__(self, width, length, terminals, discountRatio=0.98, threshold=1e-3, seed=None):
        """
        Params:
        threshold - threshold for loop termination.
        seed - random seed.
        """
        super(DPGridWorldSolver, self).__init__(
            width, length, discountRatio, seed)
        self.model = GridWorld(width, length, terminals, seed)
        self.threshold = threshold

    def policy_eval(self):
        """ DP policy evaluation. see page 74.
        """
        while True:
            diff = 0
            for (i, j) in self.policy:
                oldV = self.V[(i, j)]
                newV = sum(self._expectation_by_action((i, j)))
                self.V[(i, j)] = newV
                diff = max(diff, abs(oldV - newV))
            if diff < self.threshold:
                break

    def _expectation_by_action(self, state):
        """ helper function to compute the expecation by action in DP update.
        i.e.
            /pi(a|s) /sum_{s', r} p(s', r| s, a)[r + /gamma V(s')]
        """
        ret = []
        for action in range(len(self.policy[state])):
            reward, new_state, _ = self.model.act(action, state)
            ret.append(self.policy[state][action] *
                       (self.discountRatio * self.V[new_state] + reward))
        return ret

    def policy_improve(self):
        """ DP policy improvement. see page 76.
        """
        for (i, j) in self.policy:
            max_a = utilis.argmax(self._expectation_by_action((i, j)))
            for action in range(len(self.policy[(i, j)])):
                if action in max_a:
                    self.policy[(i, j)][action] = 1 / len(max_a)
                else:
                    self.policy[(i, j)][action] = 0


# class JackCarRentalAgent(object):
#     """ A demo Agent for Jack's Car Rental Problem in page-81.
#     """

#     def __init__(self, seed):
#         MDPAgent.__init__(self, seed)
