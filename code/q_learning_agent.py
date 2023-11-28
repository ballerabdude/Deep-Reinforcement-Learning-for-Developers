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