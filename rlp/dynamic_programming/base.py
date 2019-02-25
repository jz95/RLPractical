from rlp.mdps import MDP
from rlp.base import BaseAgent
from rlp.grid_world import GridWorld, GridWorldAgent, OFFSET
from rlp import utilis
from itertools import product


class DynamicProgrammingEnvModel(MDP):
    """ Environment for Dynamic Programming.
    """

    def __init__(self, seed=None):
        super(DynamicProgrammingEnvModel, self).__init__(seed)

    def act(self):
        print("rlp::warning:: Do not call 'act' in %s." % self.__class__)
        pass


class DynamicProgrammingAgent(BaseAgent):
    """ Agent for Dynamic Programming.
    """

    def __init__(self, seed=None):
        super(DynamicProgrammingAgent, self).__init__(seed)

    def action(self):
        print("rlp::warning:: Do not call 'action' in % s." % self.__class__)
        pass

    def update(self):
        print("rlp::warning:: Do not call 'update' in %s." % self.__class__)
        pass

    def set_experience(self, reward, new_state):
        print("rlp::warning:: Do not call 'set_experience' in %s." % self.__class__)
        pass


class DPGridWorldEnv(DynamicProgrammingEnvModel, GridWorld):
    """ Grid World Environment for Dynamic Programming.
    """

    def __init__(self, width, length, terminals, seed=None):
        GridWorld.__init__(self, width, length, terminals, seed)

    def prob_next_state_n_reward(self, state, action):
        ret = {}

        if state in self.terminal_states:
            reward = 0
            nextState = state
            ret[(nextState, reward)] = 1
            return ret

        y, x = state
        assert y >= 0 and y < self.width
        assert x >= 0 and x < self.length

        reward = -1
        dy, dx = OFFSET[action]
        y_, x_ = y + dy, x + dx
        nextState = GridWorld.clip_range(y_, 0, self.width - 1),\
            GridWorld.clip_range(x_, 0, self.length - 1)

        if nextState in self.terminal_states:
            reward = 0
        ret[(nextState, reward)] = 1

        return ret


class DPGridWorldAgent(DynamicProgrammingAgent, GridWorldAgent):
    """ Grid World Agent for Dynamic Programming.
    """

    def __init__(self, width, length, discountRatio, seed=None):
        GridWorldAgent.__init__(self, width, length, discountRatio, seed)


class JackCarRentalEnv(DynamicProgrammingEnvModel):
    """ DEMO Environment for jack's car rental problem.
    """

    def __init__(self, rental_lams, return_lams, seed=None):
        super(JackCarRentalEnv, self).__init__(seed)
        self.rental_lams = rental_lams
        self.return_lams = return_lams

    def prob_next_state_n_reward(self, state, action):
        n_car0, n_car1 = state  # num of cars avaliable at current timestep
        # move the car
        n_car0 -= action
        n_car1 += action

        ret = {}
        # customer rent and return cars
        for n_rental0, n_rental1 in product(range(0, n_car0 + 1), range(0, n_car1 + 1)):
            p_rental0 = utilis.possion_prob(n_rental0, self.rental_lams[0], truncate_threshold=n_car0)
            p_rental1 = utilis.possion_prob(n_rental1, self.rental_lams[1], truncate_threshold=n_car1)

            max_n_return0, max_n_return1 = 20 - (n_car0 - n_rental0), 20 - (n_car1 - n_rental1)
            for n_return0, n_return1 in product(range(0, max_n_return0 + 1), range(0, max_n_return1 + 1)):
                p_return0 = utilis.possion_prob(n_return0, self.return_lams[0], truncate_threshold=max_n_return0)
                p_return1 = utilis.possion_prob(n_return1, self.return_lams[1], truncate_threshold=max_n_return1)
                prob = p_rental0 * p_rental1 * p_return0 * p_return1

                n_car0_next = n_car0 - n_rental0 + n_return0
                n_car1_next = n_car1 - n_rental1 + n_return1
                nextState = n_car0_next, n_car1_next
                reward = 10 * (n_rental0 + n_rental1) - 2 * abs(action)

                # if still we get invalid value
                # just ignore this value and count it as 0
                if (nextState, reward) not in ret:
                    ret[(nextState, reward)] = prob
                else:
                    ret[(nextState, reward)] += prob

        return ret


class JackCarRentalAgent(DynamicProgrammingAgent):
    """ DEMO Agent for jack's car rental problem.
    """

    ACTIONS = list(range(-5, 6))  # predefined possbile actions

    def __init__(self, discountRatio, seed=None):
        super(JackCarRentalAgent, self).__init__(seed)
        self._init_state_value_fun()
        self._init_policy()
        self.discountRatio = discountRatio

    def _init_state_value_fun(self):
        """ initialize V function, all as 0.
        """
        self.V = {}
        for i in range(21):
            for j in range(21):
                self.V[(i, j)] = 0

    def _init_policy(self):
        """ initialize policy as a uniform distribution.
        """
        self.policy = {}
        for i in range(21):
            for j in range(21):
                if (i, j) not in self.policy:
                    self.policy[(i, j)] = {}
                possbile_actions = []
                for action in JackCarRentalAgent.ACTIONS:
                    if i - action < 0 or i - action > 20 or j + action > 20 or j + action < 0:
                        continue
                    else:
                        possbile_actions.append(action)
                for action in possbile_actions:
                    self.policy[(i, j)][action] = 1 / len(possbile_actions)