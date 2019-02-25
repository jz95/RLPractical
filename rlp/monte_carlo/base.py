from ..base import BaseEnvironment, BaseAgent
import numpy as np

DECK = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10]
CARD_PROBS = [1 / 13] * 9 + [1 / 13 * 4]
# action
HIT, STICK = 0, 1
ACTIONS = [HIT, STICK]
IN_PROGRESS, TERMINAL = 0, 1


def get_card():
    return np.random.choice(DECK, p=CARD_PROBS)


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
    cards from a infinite deck.
    """

    def __init__(self, seed):
        self._init_cards()

    def _init_cards(self):
        self.dealer_cards = [get_card() for i in range(2)]
        self.dealer_sum, _ = sum_over_cards(self.dealer_cards)

    def _judge(self):
        player_sum = state[1]

        # if the player goes bust at first, he loses
        if player_sum > 21:
            return -1, TERMINAL, None
        if player_sum == 21 and
            if self.dealer_sum == 21 and len(self.dealer_cards) == 2:
                return 0, TERMINAL, None
            else:
                return 1, TERMINAL, None

        # Dealer sticks if >= 17 else hit
        while self.dealer_sum < 17:
            self.dealer_cards.append(get_card())
            self.dealer_sum, _ = sum_over_cards(self.dealer_cards)

        # if dealer goes bust
        # and player dosen't, dealer loses
        if self.dealer_sum > 21:
            return 1, TERMINAL, None

        if self.dealer_sum == player_sum:
            return 0, TERMINAL, None
        elif self.dealer_sum > player_sum:
            return -1, TERMINAL, None
        else:
            return 1, TERMINAL, None

    def react(self, state, action):
        assert action in (HIT, STICK)

        if action == STICK:
            self._judge()
        else:
            card = get_card()

    def reset(self):
        self._init_cards()


class BlackJackAgent(BaseAgent):
    """ Black Jack Agent in page 93.
    """

    def __init__(self, seed=None):
        self._init_policy()
        self._init_state_value_fun()
        self._init_cards()

    def _init_policy(self):
        for usable_ace in (0, 1):
            for player_sum in range(12, 22):
                for dealer_show in DECK:
                    self.policy[(usable_ace, player_sum, dealer_show)] = [0.5, 0.5]

    def _init_state_value_fun(self):
        for usable_ace in (0, 1):
            for player_sum in range(12, 22):
                for dealer_show in DECK:
                    self.V[(usable_ace, player_sum, dealer_show)] = 0

    def _init_cards(self):
        self.player_cards = [get_card() for i in range(2)]
        self.player_sum, self.usable_ace = sum_over_cards(self.player_cards)

    def update(self):
        pass

    def set_experience(self, reward, new_state):
        pass

    def reset(self):
        self._init_cards()


class NaiveBlackJackAgent(BlackJackAgent):
    """ Naive agent sticks only on 20 or 21, shown in page 94.
    """

    def __init__(self, seed=None):
        super(NaiveBlackJackAgent, self).__init__(seed)

    def action(self):
        if self.player_sum in (20, 21):
            return STICK
        else:
            return HIT


class AdvancedBlackJackAgent(BlackJackAgent):
    def __init__(self, seed=None):
        super(AdvancedBlackJackAgent, self).__init__(seed)

    def action(self):
        state = (self.usable_ace, self.player_sum, self.dealer_show)
        self.random_state.choice(ACTIONS, p=self.policy[state])
