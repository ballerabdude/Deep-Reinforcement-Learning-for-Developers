# Chapter 2: Basics of Q-Learning

### Introduction to Q-Learning

Q-Learning is a foundational model-free reinforcement learning algorithm that enables agents to learn optimal policies for decision-making without knowing the dynamics of the environment they operate in. By interacting with the environment, the agent learns to associate actions with rewards and discovers the best strategy through trial and error.

### Understanding Q-values

At the core of Q-Learning are the Q-values, which represent the expected cumulative reward of taking an action in a given state and following the optimal policy thereafter. The Q-value function is denoted by Q(s, a), where 's' stands for state and 'a' for action.

### The Q-Learning Algorithm: Process and Components

The Q-Learning algorithm can be summarized in several key steps:

1. Initialize the Q-values table (Q-table) arbitrarily for all state-action pairs.
2. Observe the current state `s`.
3. Choose an action `a` for the state `s` based on a policy derived from the Q-values (e.g., ε-greedy).
4. Take the action `a`, and observe the outcome state `s'` and reward `r`.
5. Update the Q-value for the state-action pair based on the formula:
   
   Q(s, a) = Q(s, a) + α * [r + γ * maxQ(s', a') - Q(s, a)]
   
   where:
   - α (alpha) is the learning rate (0 < α ≤ 1)
   - γ (gamma) is the discount factor (0 ≤ γ < 1)
6. Set the state `s` to the new state `s'`.
7. If the end of the episode is not reached, go back to step 3.
8. Repeat these steps for many episodes to train the agent.

The components of the Q-learning algorithm are:

- **Q-table**: A lookup table where Q-values are stored for each state-action pair.
- **Policy**: A strategy that the agent employs to determine the next action based on the Q-table.
- **Learning Rate (α)**: Determines how much new information overrides old information.
- **Discount Factor (γ)**: Measures the importance of future rewards over immediate ones.
- **Reward (r)**: The signal received from the environment to evaluate the last action.

The formula Q(s, a) = Q(s, a) + α * [r + γ * maxQ(s', a') - Q(s, a)] is used to update the Q-value of the state-action pair. Let's break this down:

- Q(s, a): This is the old or previous Q-value of the pair (state, action).
- r: This is the immediate reward got after performing the action 'a' in state 's'.
- maxQ(s', a'): This is the maximum Q-value possible for the next state 's' across all possible actions. It is picked based on the current knowledge the agent has (i.e., based on the current Q-table).
- γ * maxQ(s', a'): This term is the discounted maximum future reward from the next state 's'. The discount factor 'γ' ensures the agent cares more about immediate reward over distant or future reward; if 'γ' is close to 1 the agent will consider future rewards significantly but if 'γ' is close to 0 the agent cares mostly about immediate rewards.
- [r + γ * maxQ(s', a') - Q(s, a)]: This is the difference between the learned value and the old value; it is sometimes termed as the Temporal Difference error or simply TD error.
- α * [r + γ * maxQ(s', a') - Q(s, a)]: The learning rate 'α' is applied to the TD error to moderate the update. If 'α' is closer to 1, learning is faster as it depends more on the newest experience; if 'α' is closer to 0, learning is slower as it leverages more on cumulative past experiences.

So, this entire formula represents the idea that the Q-value for a given state-action pair is updated by the learning rate 'α' times the difference between the learned value (reward plus discounted future value) and the old value. This guides the agent to better actions as it navigates and learns from its environment.

### Convergence of Q-Learning

In the context of Q-Learning, convergence refers to the point at which the Q-values stop updating significantly and remain stable. This means that the algorithm has learned the optimal policy, that is, the best action to take in each state to maximize its future rewards. 

When the Q-Learning algorithm converges, the agent's knowledge about which actions to take in the various states is as good as it can get for a given environment. From this point, running more training episodes will no longer change the agent's behavior or improve its performance.

It is important to note that convergence to the optimal Q-values assumes that proper conditions have been met. This includes assumptions like infinite visits to each state-action pair (ensuring sufficient exploration), appropriate setting of parameters such as the learning rate and discount factor, among other considerations.

---

### Project: Q-Learning in a Grid World

We'll build a Q-Learning agent to navigate a Grid World environment. This project will solidify the concepts presented in this chapter and provide practical experience with the algorithm.

#### Setting Up a Simple Grid World Environment

We'll use the Gridworld environment from Chapter 1, which provides the necessary methods to simulate an agent's interaction with the environment.

#### Implementing Q-Learning Algorithm

We will implement the Q-Learning algorithm in a Python script called `q_learning_agent.py`. The script will interface with the Gridworld class to train the agent.

#### Defining States and Actions in the Grid World

The Grid World environment will consist of discrete states and actions. The agent can move in four directions: up, down, left, and right.

#### Experimenting with Q-Value Updates

The agent will update its Q-values based on the rewards it receives from the environment. The goal is for the agent to learn how to reach the goal efficiently.

---

Now, let's begin our project with the `q_learning_agent.py` script.

```python
# q_learning_agent.py

import numpy as np
from gridworld import Gridworld
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define the Q-learning agent
class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon, action_space, state_space_size):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.action_space = action_space  # Available actions
        self.Q = np.zeros((state_space_size, len(action_space)))  # Q-value table
        

    def choose_action(self, state):
        """Choose an action using an ε-greedy policy."""
        if random.random() < self.epsilon:
            # Randomly choose an index corresponding to an action
            return random.randint(0, len(self.action_space) - 1)
        else:
            # Choose the index of the action with the highest Q-value
            return np.argmax(self.Q[state])

    def update_Q(self, state, action, reward, next_state):
        """Update the Q-value for the state-action pair using NumPy for efficient computation."""
        max_future_q = np.max(self.Q[next_state])
        current_q = self.Q[state, action]
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
        self.Q[state, action] = new_q

    def train(self, env, num_episodes):
        """Train the agent over a specified number of episodes."""
        action_space = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Actions: right, down, left, up
        total_rewards = []
        steps_per_episode = []
        
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action_index = self.choose_action(state)
                next_state, reward, done = env.step(action_space[action_index])
                self.update_Q(state, action_index, reward, next_state)
                state = next_state
                total_reward += reward
                steps += 1

            total_rewards.append(total_reward)
            steps_per_episode.append(steps)

            # Log episode information
            logger.info(f'Episode {episode + 1}/{num_episodes} - '
                        f'Total Reward: {total_reward}, '
                        f'Steps: {steps}, '
                        f'Epsilon: {self.epsilon:.4f}')
        # After training, log summary statistics
        logger.info(f'Training complete. Average reward per episode: {np.mean(total_rewards):.2f}')
        logger.info(f'Average steps per episode: {np.mean(steps_per_episode):.2f}')
		

# Main execution starts here
if __name__ == '__main__':
    # Define the environment parameters
    env_params = {
        'width': 5,
        'height': 5,
        'start': (0, 0),
        'goal': (4, 4),
        'obstacles': [(1, 1), (2, 2), (3, 3)]
    }

    # Create the Gridworld environment
    env = Gridworld(**env_params)

    # Map the actions to indices for the Q-table
    action_space = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Actions: right, down, left, up
    action_indices = {action: idx for idx, action in enumerate(action_space)}

    # Calculate the state space size
    state_space_size = env.width * env.height

    # Create the Q-learning agent with NumPy Q-table
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1, action_space=action_indices, state_space_size=state_space_size)

    # Number of episodes to train the agent
    num_episodes = 100

    # Train the agent
    agent.train(env, num_episodes)

	 # Save the trained Q-value table to a file
    np.savetxt('q_values.txt', agent.Q)

    logger.info("Training complete. Q-values saved to q_values.txt")
```

This script sets up the Q-Learning agent, defines its methods, and runs the training process. The agent is tested in the `Gridworld` environment and its Q-values are saved to a file after training.

Continuing with the remaining sections of the chapter and further description of the code:

### Running the Training Process

With the `q_learning_agent.py` script defined, we can now discuss the execution of the training process. The main objective is to run the agent through a series of episodes in the `Gridworld` environment to learn the optimal Q-values.

Training begins by initializing the environment and the agent. At the start of each episode, the environment is reset to its initial state, and the agent initializes the Q-values for that state if they haven't been seen before. The agent then repeatedly chooses actions, observes the resulting new states and rewards, and updates the Q-values until the episode ends (either the goal is reached or a terminal condition is met).

The `train` method encapsulates this process, logging the completion of each episode to help us monitor the agent's progress. After training for the specified number of episodes, the learned Q-values are saved to a file. This data can be used for analysis or to initialize the agent at a later time for further learning or policy execution.

### Analyzing the Results

After training, it's beneficial to analyze the results. You can do this by looking at the Q-values to understand what the agent has learned. Checking the convergence of Q-values can indicate whether the agent has learned a stable policy.

Another approach is to visualize the agent's behavior in the `Gridworld` by running it with the learned policy and observing its actions. This can be done by modifying the `train` method to include a rendering of the environment after each action or by creating a separate method for policy execution where the agent only selects the best-known action without further exploration.

### Considerations for Improving Learning

Several factors can influence the effectiveness of Q-Learning:
- **Learning Rate (α):** Determines how much new information overrides old information. A smaller α makes the learning updates more conservative.
- **Discount Factor (γ):** Reflects the importance of future rewards. A higher γ values future rewards more, while a lower γ results in a more myopic policy.
- **Exploration Rate (ε):** Balances exploration and exploitation. Initially, a higher ε encourages the agent to explore the environment. Over time, gradually reducing ε (known as epsilon decay) can lead to better exploitation of the learned values.
- **Initial Q-values:** Setting initial Q-values to optimistic estimates can encourage exploration. This is known as optimistic initialization.

### Conclusion and Next Steps

By implementing the Q-Learning algorithm in a practical example such as the `Gridworld`, we have gained a deeper understanding of how agents learn from their environment to make optimal decisions. With the foundation built in this chapter, we can extend our knowledge to more complex reinforcement learning algorithms and problems.

As a next step, consider experimenting with different parameters (α, γ, and ε), introducing variability in rewards, or increasing the complexity of the `Gridworld`. You can also implement enhancements to the Q-Learning algorithm, such as using experience replay or function approximation techniques for larger state spaces.

With the project and the chapter content completed, you now have a solid understanding of Q-Learning and the know-how to implement it from scratch. The next chapters will build on this knowledge and introduce you to advanced topics in reinforcement learning.

---

This concludes the full chapter and project on the basics of Q-Learning. The chapter has introduced the key concepts, algorithmic components, and practical application in a simple Grid World environment. The project has provided a hands-on experience with coding a Q-Learning agent and running it to learn a policy for navigating the Grid World.