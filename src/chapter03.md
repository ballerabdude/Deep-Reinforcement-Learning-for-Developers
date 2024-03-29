# Chapter 3: Fundamentals of Deep Learning for Reinforcement Learning

## Overview

This chapter explores the intersection of deep learning and reinforcement learning, forming the emerging field of deep reinforcement learning (DRL). Our goal is to clearly explain the fundamental concepts of deep learning and demonstrate how combining it with RL enables effective solutions for complex decision-making problems.


## Objectives

- Understand the fundamental concepts of deep learning, with a focus on neural networks.
- Learn how deep learning can be integrated with RL algorithms to improve their performance.
- Prepare to implement a deep Q-network (DQN) in the Gridworld project.

## Introduction to Neural Networks

Neural networks are designed to mimic the workings of the human brain and are made up of interconnected units called neurons. These networks have the capacity to learn from data, enabling them to perform tasks such as predictions or decision-making without the need for explicit programming.

### Key Concepts in Neural Networks

- **Neurons**: The building blocks of neural networks that take input, process it using a weighted sum followed by an activation function, and produce an output.
- **Layers**: A neural network comprises different layers of neurons, which include the input layer, one or more hidden layers, and the output layer.
- **Weights and Biases**: Parameters of the model that adjust during training to make the neural network's predictions as accurate as possible.

## How Neural Networks Learn

The process of learning in neural networks involves adjusting their weights and biases. The goal is to reduce the discrepancy between the model's predictions and the actual target values.

- **Forward Propagation**: This is the process where input data is passed through the network to generate output predictions.
- **Loss Functions**: These are measures used to assess the deviation between the network's predictions and the true outcomes.
- **Backpropagation**: A key algorithm in neural network training, backpropagation calculates the gradient of the loss function with respect to each weight and bias, using these gradients to update the parameters via gradient descent.

## Deep Learning in Reinforcement Learning

Deep learning equips RL with powerful function approximation capabilities, essential for dealing with high-dimensional state or action spaces.

### Deep Reinforcement Learning

- **Function Approximation**: The application of deep neural networks to estimate value functions or policy functions.
- **Advantages of Deep RL**: These include the ability to process complex inputs, generalize across different states, and discern underlying patterns in the data.

## Enhancing Gridworld with Deep Learning

To integrate deep learning into our Gridworld example, we will employ a deep Q-network (DQN). A DQN utilizes a neural network to estimate the Q-value function, a central concept in RL that predicts the quality of actions taken in various states.

- **Project Update**: We will implement a DQN to substitute the Q-table with a neural network that predicts Q-values for Gridworld.
- **Implementation Steps**: We outline the necessary modifications to the existing Gridworld project to incorporate a DQN.

## Summary

We recap the deep learning concepts introduced in this chapter and discuss their relevance to RL. This provides the foundation for a hands-on implementation of a DQN in the following sections.

---


## Implementing a Deep Q-Network (DQN) with PyTorch
We use PyTorch to construct our DQN. The following parts detail each section of the code.

Create a new file called `deep_q_learning_agent.py` and import the necessary libraries:

```python
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from gridworld import Gridworld  # This imports the Gridworld environment
```

### Defining the Network Architecture
We start by defining our neural network:

```python
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
```

The `SimpleMLP` class represents a multilayer perceptron with layers `fc1`, `fc2`, and `fc3`.

### DQN Agent Class
Next, we have the `DQNAgent` class, controlling action selection and learning from interactions with the environment:

```python
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
```

Here, the agent is defined with methods like `act` for decision-making and constructors for agent properties.

### Training the Model
The `train_model` function updates the network's parameters using experience replay:

```python
# Train the model based on a batch of experience
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
```

This function calculates the loss between the predicted and target Q-values and performs backpropagation to update the network weights.

### Logging Agent's Performance
Finally, a simple logging function keeps track of the agent's learning progress:

```python
# Log the performance metrics
def log_performance(episode, total_reward, steps, epsilon):
    # Print out the episode number, total reward, number of steps and the epsilon value
    print(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {steps}, Epsilon: {epsilon}")
```

This utility function prints information about the episode number, total rewards, steps, and the epsilon value.

### Example Code Execution
The main function demonstrates initializing the environment, agent, training, and logging:

```python
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

```

Within each episode, the agent selects actions, updates its Q-values, and logs its performance.

---

In this python application, we have outlined the creation of a simple feedforward neural network model using PyTorch for approximating Q-values in the Gridworld environment. The code includes a DQN agent that selects actions using an epsilon-greedy policy and stores experiences in a replay buffer. The `train_model` function updates the neural network's weights using sampled experiences to reduce the loss between predicted and target Q-values.

Please note that this example is a simplified version of DQN. For practical and more complex scenarios, additional mechanisms like experience replay buffers with more sophisticated sampling techniques and separate target networks to stabilize the Q-value predictions are recommended.