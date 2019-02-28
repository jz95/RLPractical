from rlp.base import BaseEnvironment, BaseAgent
from collections import defaultdict
from itertools import product
import numpy as np
from rlp import utilis


DECK = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10]
CARD_PROBS = [1 / 13] * 9 + [1 / 13 * 4]
# action
HIT, STICK = 0, 1
ACTIONS = [HIT, STICK]
IN_PROGRESS, TERMINAL = 0, 1


def get_card():
    ret = np.random.choice(DECK, p=CARD_PROBS)
    ret = int(ret) if ret != 'A' else ret
    return ret


def sum_over_cards(cards):
    """ return sum, usable/unusable ace 1/0
    """
    s = 0
    cnt = 0
    for card in cards:
        if card == 'A':
            cnt += 1
            s += 11
        else:
            s += card

    while s > 21 and cnt > 0:
        s -= 10
        cnt -= 1

    return s, int(cnt != 0)


class BlackJackEnv(BaseEnvironment):
    """ Environment for Black Jack Problem in page 93.
    Cards from a infinite deck.
    Player here holds at least one Ace and another card (could also be Ace)
    at the begining of each episode.
    While the Dealer hits when his sum < 17 and sticks when sum >= 17.
    """

    def __init__(self, seed=None):
        self._init_cards()

    def _init_cards(self):
        """ initialize cards for dealer and player
        """
        self.dealer_cards = [get_card() for i in range(2)]
        self.dealer_sum, _ = sum_over_cards(self.dealer_cards)

        self.player_cards = ['A', get_card()]
        self.player_sum, self.usable_ace = sum_over_cards(self.player_cards)

    def _judge(self):
        """ After player sticks, dealer draws cards and makes judgement
        """
        # if the player goes bust at first, he loses
        if self.player_sum > 21:
            return -1, None, TERMINAL
        # if the player gets natural
        if self.player_sum == 21 and len(self.player_cards) == 2:
            if self.dealer_sum == 21 and len(self.dealer_cards) == 2:
                # dealer gets natural too
                return 0, None, TERMINAL
            else:
                return 1, None, TERMINAL

        # Dealer sticks if >= 17 else hit
        while self.dealer_sum < 17:
            self.dealer_cards.append(get_card())
            self.dealer_sum, _ = sum_over_cards(self.dealer_cards)

        # if dealer goes bust
        # and player dosen't, dealer loses
        if self.dealer_sum > 21:
            return 1, None, TERMINAL

        if self.dealer_sum == self.player_sum:
            return 0, None, TERMINAL
        elif self.dealer_sum > self.player_sum:
            return -1, None, TERMINAL
        else:
            return 1, None, TERMINAL

    def react(self, state, action):
        assert action in (HIT, STICK)

        if action == STICK:
            reward, new_state, status = self._judge()
        else:
            card = get_card()
            usable_ace, player_sum, dealer_show = state

            if card == 'A':
                player_sum += 11
                if player_sum > 21:
                    player_sum -= 10
                    if usable_ace and player_sum > 21:
                        usable_ace = 0
                        player_sum -= 10
                else:
                    usable_ace = 1

            else:
                player_sum += card
                if player_sum > 21 and usable_ace:
                    player_sum -= 10
                    usable_ace = 0

            self.player_sum = player_sum
            self.usable_ace = usable_ace

            if player_sum > 21:
                reward, new_state, status = self._judge()
            else:
                reward, new_state, status = 0, (usable_ace,
                                                player_sum, dealer_show), IN_PROGRESS

        return reward, new_state, status

    def reset(self):
        """ Clear memory and restart game.
        """
        self._init_cards()
        return self.usable_ace, self.player_sum, self.dealer_cards[0]


class BlackJackAgent(BaseAgent):
    """ Abstract class for Black Jack Agent in page 93.
    """

    def __init__(self, discountRatio=0.99, seed=None):
        super(BlackJackAgent, self).__init__(seed)
        self._init_policy()

        self.discountRatio = discountRatio

        self.history_rewards = []
        self.experiences = []

        self.returns = defaultdict(list)

    def _init_policy(self):
        self.policy = {}
        for usable_ace, player_sum, dealer_show in product((0, 1), range(12, 22), DECK):
            self.policy[(usable_ace, player_sum, dealer_show)] = {}
            for action in ACTIONS:
                self.policy[(usable_ace, player_sum, dealer_show)][action] = 1 / len(ACTIONS)

    def set_state(self, state):
        self.curr_state = state

    def reset(self):
        self.experiences.clear()
        self.history_rewards.clear()


class NaiveBlackJackAgent(BlackJackAgent):
    """ Naive agent sticks only on 20 or 21, shown in page 94.
    """

    def __init__(self, discountRatio=0.99, seed=None):
        super(NaiveBlackJackAgent, self).__init__(discountRatio, seed)
        self._initVs()

    def _initVs(self):
        self.V = {}
        for usable_ace, player_sum, dealer_show in product((0, 1), range(12, 22), DECK):
            self.V[(usable_ace, player_sum, dealer_show)] = 0

    def action(self):
        player_sum = self.curr_state[1]
        if player_sum in (20, 21):
            return STICK
        else:
            return HIT

    def update(self):
        """ update V function after each episode
        """
        T = len(self.experiences)
        G = 0
        for t in range(T - 1, -1, -1):
            state = self.experiences[t]
            reward = self.history_rewards[t]
            if state in self.experiences[:t - 1]:
                G = self.discountRatio * G + reward
            else:
                self.returns[state].append(G)
                self.V[state] = np.mean(self.returns[state])

    def set_experience(self, reward, new_state):
        self.history_rewards.append(reward)
        self.experiences.append(self.curr_state)


class AdvancedBlackJackAgent(BlackJackAgent):
    """ An Advanced Black Jack agent.
    """

    def __init__(self, discountRatio=0.99, seed=None):
        super(AdvancedBlackJackAgent, self).__init__(discountRatio, seed)
        self._initQs()
        self.is_start = True

    def _initQs(self):
        self.Q = {}
        for usable_ace, player_sum, dealer_show in product((0, 1), range(12, 22), DECK):
            self.Q[(usable_ace, player_sum, dealer_show)] = {}
            for action in ACTIONS:
                self.Q[(usable_ace, player_sum, dealer_show)][action] = 0

    def action(self):
        # keep exploring start
        if self.is_start:
            self.action_ = self.random_state.choice(ACTIONS)
            self.is_start = False
        else:
            probs = [self.policy[self.curr_state][act] for act in ACTIONS]
            self.action_ = self.random_state.choice(ACTIONS, p=probs)
        return self.action_

    def update(self):
        T = len(self.experiences)
        G = 0
        for t in range(T - 1, -1, -1):
            state, action = self.experiences[t]
            reward = self.history_rewards[t]
            if (state, action) in self.experiences[:t - 1]:
                G = self.discountRatio * G + reward
            else:
                self.returns[(state, action)].append(G)
                self.Q[(state, action)] = np.mean(
                    self.returns[(state, action)])

                opt_acts = utilis.argmax(self.Q[state])

                for action in self.Q[state]:
                    if action == opt_acts[0]:
                        self.policy[state][action] = 1
                    else:
                        self.policy[state][action] = 0

    def set_experience(self, reward, new_state):
        self.history_rewards.append(reward)
        self.experiences.append((self.curr_state, self.action_))

    def reset(self):
        super(AdvancedBlackJackAgent, self).reset()
        self.is_start = True
