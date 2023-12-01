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