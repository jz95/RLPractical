from rlp.monte_carlo.base import *


if __name__ == '__main__':
    # agent = NaiveBlackJackAgent()
    agent = AdvancedBlackJackAgent()
    env = BlackJackEnv()

    for episode in range(1000):
        state = env.reset()
        agent.reset()
        status = IN_PROGRESS
        while status != TERMINAL:
            agent.set_state(state)
            action = agent.action()
            reward, state, status = env.react(state, action)
            agent.set_experience(reward, state)
        agent.update()
