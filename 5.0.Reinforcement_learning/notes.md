# Reinforced Learning

- Data are in state/action pairs
- Goal: maximise rewards
- example: agent eat a apple to gain lifes

## Key concepts

1. Agent do some **action** to the environment
2. environment provides **observations**<u>(state)</u> back to agent
3. repeat step 1,2
4. goal of agent to get reward, not necessarily rewarded immediately

### Rewards

Total Reward (return) = reward(t) + reward(t+1) + reward(t+n)

![](discounted-reward.png)

- discounted so that agent chooses most present rewards
- reward is lesser more into the future

### Q-function

![q-function formula](q-function.png)

- to know that whether that particular action in the state is maximising the reward

#### Strategy

- enter all states, and check which q-value is highest -> then execute

![](strategy.png)

- Strategy: choose higher value action

## Algorithim

### Value Learning

- Get Q function and infer best policy

#### Deep Q Network(DQN)

### Policy Learning

- directly learn policy
