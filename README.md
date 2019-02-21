# RLPractical
This python library is meant to help the students choosing [Reinforcement Learning](https://www.inf.ed.ac.uk/teaching/courses/rl/) in University of Edinburgh. Basic algorithms mentioned in the lecture or on the [textbook](http://incompleteideas.net/book/the-book-2nd.html) would be implemented. **Note: It's still an unstable version. API might change later on.** And Docs are still incompelte.

Clone this repo and install rlp.
```
git clone https://github.com/JZ95/RLPractical.git
cd RLPractical
python setup.py install
```
or use develop mode:
```
python setup.py develop
```

open python shell, and type the following command to test if install successfully
```
import rlp
```
-------------
## Example
### 10-Armed Bandit DEMO
```
from rlp.multi_armed_bandits.envs import MultiArmedBandit
from rlp.multi_armed_bandits.agents import EpsGreedy

means = np.random.normal(0, 1, 10)               # assign mean rewards for 10 arms
stds = np.ones(10)                               # assign std for 10 arms
initQ = np.zeros(10)                         	 # assign inital action-value estimates Q0

bandit = MultiArmedBandit(k=10, means=means, stds=stds)
agent = EpsGreedy(eps=0.1, Q0=initQ)             # build eps(0.1)-greedy agent

# run agent 1000 timesteps
for _ in range(1000):
    At = agent.action()                          # agent selects action
    Rt = bandit.reward(At)                       # environment gives reward based on agent's action
    agent.get_reward(Rt)                         # agent reveives reward and updates status
    agent.update()
```

### Finite Markov Decision Process DEMO
```
TBD
```


-------------
## Run Notebooks
Please install jupyter first.
After installing jupyter and rlp, open NoteBook Server.
```
jupyter notebook
```
Open an another shell and
```
cd notebooks
```
See the notebooks and have fun with tuning params.

![ScreenShot](./imgs/readme_img1.png)


🍺 ENJOY!
