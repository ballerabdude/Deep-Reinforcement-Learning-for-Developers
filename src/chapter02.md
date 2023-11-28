# Chapter 2: Introduction to Q-Learning

## Part 1: Theoretical Foundations

### Overview

This chapter explores the concept of Q-Learning, a model-free reinforcement learning algorithm that enables agents to learn optimal action-value functions and derive an optimal policy.

### Objectives

- Introduce the Q-Learning algorithm and its role in reinforcement learning.
- Understand the components and operation of the Q-Learning algorithm.
- Prepare for the practical application of Q-Learning in the Gridworld project.

### What is Q-Learning?

Q-Learning is a method that allows agents to learn the value of an action in a particular state, guiding them to make optimal decisions through trial and error.

#### Key Concepts of Q-Learning

- **Q-value (Q-function)**: An estimation of the total expected rewards an agent can get, given a state and action.
- **Q-table**: A table where Q-values for each state-action pair are stored.

### Why Q-Learning?

Q-Learning is advantageous as it can handle environments with stochastic transitions and rewards without requiring a model of the environment.

### The Q-Learning Algorithm

The Q-Learning algorithm updates the Q-values for each state-action pair using the Bellman equation iteratively:
$$ Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)] $$
where:
- \( s, a \): current state and action
- \( s' \): next state
- \( r \): reward received
- \( \alpha \): learning rate
- \( \gamma \): discount factor
- \( a' \): possible next action


### Coding Q-Learning Components

To implement Q-Learning in our Gridworld project, we will create the Q-table and implement the update rule as part of the agent's learning algorithm.

### Summary

In this section, we've introduced the theoretical underpinnings of Q-Learning, setting the stage for a practical implementation in the Gridworld environment.

## Part 2: Practical Implementation

### Implementing Q-Learning in Gridworld

To extend our Gridworld project with the Q-Learning algorithm, we will add functionality to our agent to learn from the environment using a Q-table.

#### Q-table Initialization

We initialize a Q-table that maps each state-action pair to its Q-value, which will be updated as the agent learns.

#### Q-Learning Update

After taking an action, we update the Q-table entries using the Q-Learning formula, which incorporates the reward received and the estimated value of the next state.

#### Exploration vs. Exploitation

We'll implement an &epsilon;-greedy strategy, enabling the agent to explore the environment while exploiting its current knowledge.

### Enhancing the Agent with Q-Learning

We upgrade our `RandomAgent` to a `QLearningAgent` that can choose actions based on learned Q-values and improve its decisions over time.

#### File: `q_learning_agent.py`

```python
# q_learning_agent.py
import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_space, action_space, alpha, gamma, epsilon, action_mapping):
        self.q_table = np.zeros((state_space, action_space))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space = action_space
        self.action_mapping = action_mapping  # Add this line to store the action mapping

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action_index = random.randint(0, self.action_space - 1)  # Explore action space
        else:
            action_index = np.argmax(self.q_table[state])  # Exploit learned values
        
        # Return the action index instead of the action itself
        return action_index

    def update_q_table(self, current_state, action_index, reward, next_state):
        # Find the best action at the next state
        best_next_action = np.argmax(self.q_table[next_state])

        # Calculate the target for the Bellman equation
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]

        # Calculate the difference between the target and the current Q-value
        td_delta = td_target - self.q_table[current_state, action_index]

        # Update the Q-value for the current state-action pair
        self.q_table[current_state, action_index] += self.alpha * td_delta
```

#### Running the Q-Learning Agent

We will simulate the `QLearningAgent` within the Gridworld, observing how it learns from experiences to improve its navigation strategy.

#### File: `gridworld_q_learning.py`

```python
# gridworld_q_learning.py
from gridworld import Gridworld
from q_learning_agent import QLearningAgent

# Define the Gridworld environment dimensions and attributes
grid_width = 5
grid_height = 5
num_states = grid_width * grid_height
num_actions = 4  # Up, down, left, and right

# Define action mapping that maps action indices to actual actions (movements)
action_mapping = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left

# Initialize the Gridworld environment
gridworld = Gridworld(width=grid_width, height=grid_height, start=(0, 0), goal=(4, 4), obstacles=[(1, 1), (2, 2), (3, 3)])

# Initialize the Q-Learning agent with hyperparameters alpha, gamma, and epsilon
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
agent = QLearningAgent(state_space=num_states, action_space=num_actions, alpha=alpha, gamma=gamma, epsilon=epsilon, action_mapping=action_mapping)

# Run the Q-Learning agent in the Gridworld
num_episodes = 20  # Total number of episodes to run
for episode in range(num_episodes):
    done = False
    total_reward = 0
    gridworld.reset()
    current_state = gridworld.get_state()

    print(f"Starting episode {episode + 1}")

        

    while not done:
        
        # Agent chooses an action index
        action_index = agent.choose_action(current_state)

        # Perform the chosen action and get the result
        _, reward, done = gridworld.step(agent.action_mapping[action_index])

        # Convert the next position to the next state index
        next_state = gridworld.get_state()  # Assuming Gridworld has this method

        # Update the Q-table with the action index
        agent.update_q_table(current_state, action_index, reward, next_state)

        # Update total reward and current state
        total_reward += reward
        current_state = next_state

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

```

### Summary

We've now set the theoretical foundation and implemented the Q-Learning algorithm in our Gridworld agent. The agent's ability to learn from interactions with the environment will be demonstrated in the following simulation.

## Part 3: Updating the Gridworld Environment

### Enhancing State Representation

To facilitate our Q-Learning agent's interaction with the Gridworld, we must convert the environment's grid coordinates into a state index.

### Objectives

- Enable the Gridworld to provide a state index representation for the Q-Learning agent.
- Update the Gridworld class with a `get_state()` method.

### The `get_state()` Method

We add a method to the `Gridworld` class that will map the agent's 2D grid coordinates to a unique state index.

### Implementation

We will implement the `get_state()` method to provide a state representation by flattening the grid.

#### File: `gridworld.py`

```python
class Gridworld:
    # Existing methods...

    def get_state(self):
        """
        Returns a unique state index for the agent's current position.
        """
        x, y = self.agent_position
        return y * self.width + x
```

### Summary and Integration

The `Gridworld` class now provides a state index that our Q-Learning agent can use to update the Q-table and make decisions. This integration is crucial for the upcoming simulations where the agent learns to navigate the environment more effectively.

### Next Steps

With state representation in place, we can proceed to initialize the Q-table and run the Q-Learning agent through training episodes to observe its learning and performance improvements. The next sections of this chapter will focus on these aspects, leading to a fully functional Q-Learning agent in the Gridworld.