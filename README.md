# Monte Carlo First Visit and Every Visit Estimates for GridWorld

Monte Carlo methods require only experience—sample sequences of states, actions, and rewards from actual or simulated interaction with an environment. Here we do not assume complete knowledge of the environment. Monte Carlo methods are ways of solving the reinforcement learning problem based on averaging sample returns. Monte Carlo methods sample and average returns for each state–action pair. Here, we learn value functions from sample returns with the MDP.

Value of a state is estimated it from experience, by averaging the returns observed after visits to that state. As more returns are observed, the average should converge to the expected value. The Monte Carlo Prediction methods are of two types: First Visit Monte Carlo Method and Every Visit Monte Carlo Method. The first-visit MC method estimates v<sub>π</sub>(s) as the average of the returns following first visits to s, whereas the every-visit MC method averages the returns following all visits to s.

## MC Algortihm

![image](https://user-images.githubusercontent.com/22128902/92507603-db499b00-f224-11ea-8062-9748c0472e3c.png)

To estimate the value of the states by MC Every Visit Method, check for St having occurred earlier in the episode is eliminated

## Grid World Problem

![image](https://user-images.githubusercontent.com/22128902/92507684-f6b4a600-f224-11ea-9af2-df0ac421f97a.png)

We are to find the Monte Carlo first visit and every visit estimates for all the states of a 4*4 grid world following the below optimal policy:

![image](https://user-images.githubusercontent.com/22128902/92507750-12b84780-f225-11ea-92b0-45b1701fd10e.png)

The reward is -1 on all transitions until the terminal state is reached.
Also the probability of taking the optimal action is 0.7 while the non-optimal actions have 0.1 probability respectively. In a state where multiple optimal actions are feasible the optimal action is chosen with equal probability. Due to this probability distribution, it is observed that the values of the all the states (computed later on), are more than the expected value of the states computed for the random policy which leads to the optimal policy which is:

![image](https://user-images.githubusercontent.com/22128902/92507794-295e9e80-f225-11ea-9779-3b2380b67fdb.png)

![image](https://user-images.githubusercontent.com/22128902/92507824-367b8d80-f225-11ea-88fd-84f1880ccbfe.png)


