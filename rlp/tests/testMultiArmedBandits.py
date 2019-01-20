import numpy as np
from rlp.envs import MultiArmedBandit
from rlp.agents import UCB, GradientBandit, EpsGreedy
import matplotlib.pyplot as plt
from tqdm import tqdm


def getCumOptActRate(actions, optimal):
    """get cumulative optimal action rate.
    ===================
    params:
    actions - action list given by agent
    optimal - optimal action defined by env
    """
    hitOpt = (np.array(actions) == optimal)
    return np.cumsum(hitOpt) / np.cumsum(np.ones_like(hitOpt))


def getCumAvgRewards(rewards):
    """get cumulative average rewards.
    ===================
    params:
    rewards - rewards list given by agent
    """
    return np.cumsum(rewards) / np.cumsum(np.ones_like(rewards))


def draw(stats):
    _, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 8))
    ax = ax.ravel()
    for i in range(2):
        for conf in stats:
            ax[i].plot(stats[conf][i], label=conf)

    ax[0].set_ylabel('average reward')
    ax[1].set_ylabel('optimal action rate')
    plt.xlabel('steps')
    plt.tight_layout()
    plt.legend()
    plt.show()


def runGreedy(n_timesteps, eps):
    means = np.random.normal(0, 1, 5)
    stds = np.ones(5)
    # assign inital action value estimates Q0 as 5 (over-optimistic)
    initQ = np.zeros((5, ))

    bandit = MultiArmedBandit(k=5, means=means, stds=stds)
    agent = EpsGreedy(eps=0.1, Q0=initQ)
    for _ in range(n_timesteps):
        # pdb.set_trace()
        At = agent.action()
        print('choose ', At)
        Rt = bandit.reward(At)
        agent.get_reward(Rt)
        agent.update()


def runUCB(n_timesteps, c):
    means = np.random.normal(0, 1, 10)
    stds = np.ones(10)
    # assign inital action value estimates Q0 as 5 (over-optimistic)
    initQ = np.zeros((10, ))

    bandit = MultiArmedBandit(k=10, means=means, stds=stds)
    agent = UCB(alpha=0.2, conf_level=c, Q0=initQ)
    for _ in range(n_timesteps):
        # pdb.set_trace()
        At = agent.action()
        Rt = bandit.reward(At)
        agent.get_reward(Rt)
        agent.update()

    cum_rewards = getCumAvgRewards(agent.rewards)
    optimal_action_rate = getCumOptActRate(agent.actions, np.argmax(means))
    return cum_rewards, optimal_action_rate, agent


def runGB(n_timesteps):
    means = np.random.normal(5, 1, 10)
    stds = np.ones(10)
    # assign inital action value estimates Q0 as 5 (over-optimistic)
    initH = np.random.randn(10)

    bandit = MultiArmedBandit(k=10, means=means, stds=stds)
    agent = GradientBandit(H0=initH, alpha=0.2)
    for _ in range(n_timesteps):
        # pdb.set_trace()
        At = agent.action()
        Rt = bandit.reward(At)
        agent.get_reward(Rt)
        agent.update()


def runUCB(n_timesteps, isUCB):
    means = np.random.normal(0, 1, 10)
    stds = np.ones(10)
    initQ = np.zeros(10)

    bandit = MultiArmedBandit(k=10, means=means, stds=stds)
    agent = UCB(alpha=0.2, conf_level=5, Q0=initQ) if isUCB else EpsGreedyConstStep(
        alpha=0.2, eps=0.1, Q0=initQ)
    for _ in range(n_timesteps):
        At = agent.action()
        Rt = bandit.reward(At)
        agent.get_reward(Rt)
        agent.update()

    cum_rewards = getCumAvgRewards(agent.rewards)
    optimal_action_rate = getCumOptActRate(agent.actions, np.argmax(means))

    return cum_rewards, optimal_action_rate


def main():
    nRuns = 100
    nTimeStep = 5000
    UCBStats = {}
    cumRewardMat = np.zeros((nRuns, nTimeStep))
    cumOptActRateMat = np.zeros((nRuns, nTimeStep))

    settings = {
        'UCB c=2': True,
    }

    for conf, isUCB in tqdm(settings.items(), desc='loop over settings'):
        cumRewardMat = np.zeros((nRuns, nTimeStep))
        cumOptActRateMat = np.zeros((nRuns, nTimeStep))
        for i in tqdm(range(nRuns), desc='repeat exps', leave=False):
            cumRewardMat[i], cumOptActRateMat[i] = runUCB(nTimeStep, isUCB)
        UCBStats[conf] = (cumRewardMat.mean(axis=0),
                          cumOptActRateMat.mean(axis=0))

    draw(UCBStats)


if __name__ == '__main__':
    main()
