## Bellman equations
They are the consistency conditions that the value functions under a given policy must satisfy. By solving Bellman equations, we can get the value fuctions for the given policy.

## Optimality issues
- The optimal value functions assign to each state, or state–action pair, the **largest** expected return achievable by any policy.
- A policy whose value functions are optimal is an optimal policy.
- Any policy that is greedy w.r.t the **optimal** value functions must be an optimal policy.

## Bellman optimality equations
- They define the special consistency conditions that the **optimal** value functions must satisfy. In principle, they can be solved for the optimal value functions, from which an optimal policy can be determined.
- In practical, solving the Bellman optimality equations is infeasible, since they rely on 3 assumptions: 
    - we accurately know the dynamics of the environment; 
    - we have enough computational resources to complete the computation of the solution; 
    - the Markov property.
- Therefore we develop a series of action-value based methods (MC, TD, etc) to approximate tp the solution of Bellman optimality equations.