from .base import BaseEnvironment, BaseAgent
"""
This module provides basic API for the simple gridworld example.
"""

# ACTION ID
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
# ACTION OFFSET
OFFSET = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
# STATUS
IN_PROGRESS, TERMINAL = 0, 1


class GridWorld(BaseEnvironment):
    """ Environment for Grid World problem,
    where reward is fixed as -1 at each step,
    except the transition into terminal state, which is 0.
    """

    def __init__(self, width, length, terminals, seed):
        """Params:
        width - int, number of cells in width, vertical direction.
        length - int, number of cells in length, honrizontal direction.
        terminals - list of tuples, coordinates of terminial state.
        seed - random seed.
        """
        self.width = width
        self.length = length
        self.terminal_states = terminals
        super(GridWorld, self).__init__(seed)

    def react(self, state, action):
        """Params:
        state - tuple, representing coordinate in grid world。
        action - int, action denoted the movement of agent at one step。
        =====================
        Return:
        reward, new_state, status
        """
        if state in self.terminal_states:
            return 0, state, TERMINAL

        y, x = state
        assert y >= 0 and y < self.width
        assert x >= 0 and x < self.length

        reward = -1
        dy, dx = OFFSET[action]
        y_, x_ = y + dy, x + dx
        new_state = GridWorld.clip_range(y_, 0, self.width - 1),\
            GridWorld.clip_range(x_, 0, self.length - 1)

        status = int(new_state in self.terminal_states)
        if status:
            reward = 0

        return reward, new_state, status

    @staticmethod
    def clip_range(x, low_bnd, up_bnd):
        """ clip x according to the given low_bnd and up_bnd
        e.g. given low_bnd = 0, up_bnd = 3,
        if x = -1, return 0
        if x = 5, return 3
        """
        assert low_bnd <= up_bnd
        return min(max(x, low_bnd), up_bnd)

    def __repr__(self):
        return 'GridWorld-Environment (%d, %d)\n\t start (%d, %d),\
                     end (%d, %d)' % (self.width, self.length)


class GridWorldAgent(BaseAgent):
    """Agent for Grid World Case.
    """

    def __init__(self, width, length, discountRatio, seed):
        """
        Params:
        width - number of finite states.
        length - number of finite actions.
        discountRatio - discount ratio for returns.
        seed - random seed.
        """
        super(GridWorldAgent, self).__init__(seed)
        self.width = width
        self.length = length
        self.discountRatio = discountRatio

        self._init_action_value_fun()
        self._init_state_value_fun()
        self._init_policy()

    def _init_state_value_fun(self):
        """ initialize V function, all as 0.
        """
        self.V = {}
        for i in range(self.width):
            for j in range(self.length):
                self.V[(i, j)] = 0

    def _init_action_value_fun(self):
        """ initialize Q function, all as 0.
        """
        self.Q = {}
        for i in range(self.width):
            for j in range(self.length):
                self.Q[(i, j)] = dict([(a, 0) for a in ACTIONS])

    def _init_policy(self):
        """ initialize policy as a uniform distribution.
        """
        self.policy = {}
        for i in range(self.width):
            for j in range(self.length):
                self.policy[(i, j)] = dict([(a, 1 / len(ACTIONS)) for a in ACTIONS])

    def action(self):
        """ Agent make action.
        """
        probs = [self.policy[self.currState][a] for a in ACTIONS]
        return self.random_state.choice(ACTIONS, probs)
