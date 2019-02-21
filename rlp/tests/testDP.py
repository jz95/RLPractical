from rlp.dynamic_programming.agents import DPGridWorldSolver


def main():
    agent = DPGridWorldSolver(5, 5, [(0, 0), (5, 5)])
    for _ in range(5):
        agent.policy_eval()
        agent.policy_imporve()


if __name__ == '__main__':
    main()