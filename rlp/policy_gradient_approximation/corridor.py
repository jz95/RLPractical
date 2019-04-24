import numpy as np
from ..base import BaseEnvironment, BaseAgent

RIGHT, LEFT = 0, 1
ACTIONS = [RIGHT, LEFT]
IN_PROGRESS, TERMINAL = 0, 1


class Corridor(BaseEnvironment):
    """ DEMO env for Short corridor with switched actions, see page 323
    """

    def __init__(self, seed):
        super(self, Corridor).__init__(seed)

    def step(self, state, action):
        assert action in ACTIONS
        if state == 0:
            if action == LEFT:
                return -1, state, IN_PROGRESS
            else:
                return -1, 1, IN_PROGRESS

        if state == 1:
            # the swith poistion
            if action == LEFT:
                return -1, 1, IN_PROGRESS
            else:
                return -1, 0, IN_PROGRESS

        if state == 2:
            if action == RIGHT:
                return 0, 3, TERMINAL
            else:
                return -1, 1, IN_PROGRESS


class CorridorAgent(BaseAgent):
    def __init__(self, seed):
        super(self, Corridor).__init__(seed)
        self.theta = np.rand.randn(len(ACTIONS))

    def _init_policy(self):
        pass

    def _process_state_action(self, state, action):
        """ return the vector representation for
        the state and action pair.
        """

    def action(self):
        probs = [self.policy[self.currState][a] for a in ACTIONS]
        return self.random_state.choice(ACTIONS, probs)

    def update(self):
        pass
