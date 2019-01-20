# RLPractical
This python library is meant to help the students choosing Reinforcement Learning in University of Edinburgh. Basic Algorithms presented in lecture or on the [textbook](http://incompleteideas.net/book/the-book-2nd.html) would be implemented. **Note: It's still an unstable version. API migh change later on.** And Docs are still incompelte.

install:
```
python setup.py install
```
or use develop mode:
```
python setup.py develop
```

open python shell, and type
```
import rlp
```
-------------
## Usage
```
from rlp.envs import MultiArmedBandit
from rlp.agents import EpsGreedy

means = np.random.normal(0, 1, 10)               # assign mean rewards for 10 arms
stds = np.ones(10)                               # assign std for 10 arms
initQ = np.zeros(10)                         # assign inital action-value estimates Q0

bandit = MultiArmedBandit(k=10, means=means, stds=stds)
agent = EpsGreedy(eps, Q0=initQ)              # build eps-greedy agent

# run agent 1000 timesteps
for _ in range(1000):
    At = agent.action()                       # agent selects action
    Rt = bandit.reward(At)                    # environment gives reward based on agent's action
    agent.get_reward(Rt)                      # agent reveives reward and updates status
    agent.update()
```



-------------
## Run Notebooks
Please install jupyter first.
After installing jupyter and rlp, in cmd type 
```
jupyter notebook
cd notebooks
```
See the notebooks and have fun with tuning params.
