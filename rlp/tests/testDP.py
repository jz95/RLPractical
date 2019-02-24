from rlp.dynamic_programming.solver import JackCarRentalSolver, DPGridWorldSolver
from rlp.dynamic_programming.base import JackCarRentalAgent, JackCarRentalEnv, DPGridWorldAgent, DPGridWorldEnv
from rlp import utilis

def test_jack_car_rental():
    agent = JackCarRentalAgent(discountRatio=0.9)
    env_model = JackCarRentalEnv((3, 4), (3, 2))
    solver = JackCarRentalSolver(agent, env_model, 0.1)

    for k in range(1):
        solver.policy_eval(onestep=True)
        solver.policy_improve()
        values = solver.agent.V
        policy = solver.agent.policy


def test_possion():
    # print(utilis.possion_prob(1, 2))
    # print(utilis.possion_prob(2, 2))
    print(utilis.possion_prob(0, 2, truncate_threshold=0))
    print(utilis.possion_prob.memo)
    print(utilis.possion_prob.factorial)

def main():
    # test_possion()
    test_jack_car_rental()

if __name__ == '__main__':
    main()