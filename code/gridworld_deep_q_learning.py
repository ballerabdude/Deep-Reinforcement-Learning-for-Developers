import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from gridworld import Gridworld  # This imports the Gridworld environment

# Define the neural network architecture for Deep Q-Learning
class SimpleMLP(nn.Module):
    # Constructor for the neural network
    def __init__(self, input_size, output_size, hidden_size=64):
        super(SimpleMLP, self).__init__()
        # First fully connected layer from input_size to hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Second fully connected layer, hidden to hidden
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Third fully connected layer from hidden_size to output_size
        self.fc3 = nn.Linear(hidden_size, output_size)

    # Forward pass definition for the neural network
    def forward(self, x):
        # Apply ReLU activation function after first layer
        x = F.relu(self.fc1(x))
        # Apply ReLU activation function after second layer
        x = F.relu(self.fc2(x))
        # Output layer, no activation function
        return self.fc3(x)

# Deep Q-Network agent class definition
class DQNAgent:
    # Constructor for the DQN agent
    def __init__(self, input_size, output_size, hidden_size=64):
        # The neural network model
        self.model = SimpleMLP(input_size, output_size, hidden_size)
        # Epsilon for the epsilon-greedy policy (initially set to 1 for full exploration)
        self.epsilon = 1.0
        # Minimum value that epsilon can decay to over time
        self.epsilon_min = 0.01
        # Rate at which epsilon is decayed over time
        self.epsilon_decay = 0.995
        # List to hold past experiences for replay
        self.memory = []

    # Method to decide an action based on the current state
    def act(self, state):
        # Check if we should take a random action (exploration)
        if np.random.rand() <= self.epsilon:
            # Return a random action within the action space size
            return random.randrange(output_size)
        
        # If not exploring, process the state through the DQN to get the action values
        # First, ensure state is a PyTorch tensor
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().unsqueeze(0)
        
        # Forward pass through the network to get action values
        action_values = self.model(state)
        # Return the action with the highest value
        return np.argmax(action_values.detach().numpy())

    # Method to store experiences in replay memory
    def remember(self, state, action, reward, next_state, done):
        # Append the experience as a tuple to the memory list
        self.memory.append((state, action, reward, next_state, done))

    # Method to decay epsilon over time for less exploration
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Function to train the neural network model
def train_model(agent, optimizer, batch_size, gamma):
    # Check that there are enough experiences in memory to sample a batch
    if len(agent.memory) < batch_size:
        return
    # Sample a minibatch of experiences from memory
    minibatch = random.sample(agent.memory, batch_size)
    # Unpack the experiences
    states, actions, rewards, next_states, dones = zip(*minibatch)

    # Convert experience components to PyTorch tensors
    states = torch.from_numpy(np.vstack(states)).float()
    actions = torch.from_numpy(np.vstack(actions)).long()
    rewards = torch.from_numpy(np.vstack(rewards)).float()
    next_states = torch.from_numpy(np.vstack(next_states)).float()
    dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

    # Calculate the expected Q values from the neural network
    Q_expected = agent.model(states).gather(1, actions)
    # Calculate the Q value for the next states and get the max Q value for each next state
    Q_targets_next = agent.model(next_states).detach().max(1)[0].unsqueeze(1)
    # Calculate the target Q values for the current states using the Bellman equation
    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

    # Calculate the loss between the expected Q values and the target Q values
    loss = F.mse_loss(Q_expected, Q_targets)
    # Backpropagate the loss, update the network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Function to log the performance of the agent
def log_performance(episode, total_reward, steps, epsilon):
    # Print out the episode number, total reward, number of steps and the epsilon value
    print(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {steps}, Epsilon: {epsilon}")

# Example usage of the defined classes and functions
if __name__ == "__main__":
    # Initialize the Gridworld environment
    gridworld = Gridworld(width=5, height=5, start=(0, 0), goal=(4, 4), obstacles=[(1, 1), (2, 2), (3, 3)])
    # Define action mapping that maps action indices to movements
    action_mapping = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
    # Calculate the input size based on the environment's state space
    input_size = gridworld.width * gridworld.height
    # The output size is the number of possible actions
    output_size = 4
    # Initialize the DQN agent
    agent = DQNAgent(input_size, output_size)
    # Define an optimizer for the neural network (Adam optimizer with a learning rate of 0.001)
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=0.001)
    # Batch size for experience replay
    batch_size = 32
    # Discount factor for future rewards
    gamma = 0.99

    # Function to convert a state to a tensor for neural network input
    def state_to_tensor(state, grid_size=5):
        # Create a one-hot encoded vector for the state
        state_vector = torch.zeros(grid_size * grid_size, dtype=torch.float32)
        state_vector[state] = 1
        return state_vector

    # Main training loop
    for episode in range(100):
        # Reset the environment at the start of each episode
        state = gridworld.reset()
        # Initialize total reward and steps counter
        total_reward = 0
        done = False
        steps = 0

        # Loop for each step in the episode
        while not done:
            # Convert state to tensor format
            state_vector = state_to_tensor(state)
            # Select an action using the DQN agent's policy
            action = agent.act(state_vector)
            # Take the action in the environment and observe the next state and reward
            next_state, reward, done = gridworld.step(action_mapping[action])
            # Convert the next state to tensor format
            next_state_vector = state_to_tensor(next_state)
            # Remember the experience
            agent.remember(state_vector, action, reward, next_state_vector, done)

            # Move to the next state
            state = next_state
            # Update the total reward
            total_reward += reward
            # Increment the step counter
            steps += 1

            # Train the model with experiences from memory
            train_model(agent, optimizer, batch_size, gamma)

        # After the episode, decay epsilon for less exploration in future episodes
        agent.update_epsilon()
        # Log the performance metrics for the episode
        log_performance(episode, total_reward, steps, agent.epsilon)