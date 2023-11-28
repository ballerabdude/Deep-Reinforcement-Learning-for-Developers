# Chapter 1: Introduction to Reinforcement Learning

## Part 1: Understanding the Fundamentals

### Overview
Reinforcement learning (RL) is a paradigm in machine learning that provides a framework for agents to learn how to behave in an environment by performing actions and seeing the results. This chapter introduces RL, explores its key concepts, and applies them in a simple Gridworld environment.

### Objectives
- Understand the core principles of RL.
- Identify RL applications in different industries.
- Establish a foundational project in RL.

### What is Reinforcement Learning?
RL involves an agent that learns to make decisions by taking actions in an environment to maximize some notion of cumulative reward. It's characterized by trial and error, feedback loops, and adaptability to changing situations.

#### Key Components of RL
- **Agent**: The learner or decision-maker.
- **Environment**: Where the agent takes actions.
- **Action**: A set of operations that the agent can perform.
- **State**: The current situation of the environment.
- **Reward**: Feedback from the environment to the agent.

### Why Reinforcement Learning?
Reinforcement learning is unique in its approach to problem-solving, allowing agents to learn from their own experiences rather than being told the correct actions. This is particularly useful in complex, unpredictable environments or when the desired behavior is difficult to express with explicit rules.

### Applications of RL
- **Game playing**: Achieving superhuman performance in complex games.
- **Robotics**: Teaching robots to perform tasks autonomously.
- **Autonomous vehicles**: Driving safely and efficiently in dynamic environments.

### Project Overview: The Gridworld
The Gridworld is an introductory RL project where an agent must navigate through a grid to reach a goal while avoiding obstacles.

#### Project Goal
Create an agent that finds the shortest path to a goal within a grid, considering obstacles.

#### Key Learning
This project will teach you about the interaction between an agent and its environment, the role of rewards, and how to implement these concepts in code.

### Coding the Gridworld Environment
We construct a `Gridworld` class in Python to simulate the environment for our agent.

- File: `gridworld.py`

```python
# gridworld.py
import numpy as np

class Gridworld:
    def __init__(self, width, height, start, goal, obstacles):
        """
        Initializes a Gridworld object.

        Parameters:
        - width (int): The width of the grid.
        - height (int): The height of the grid.
        - start (tuple): The starting position of the agent as a tuple (x, y).
        - goal (tuple): The goal position of the agent as a tuple (x, y).
        - obstacles (list): A list of obstacle positions as tuples [(x1, y1), (x2, y2), ...].
        """
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.reset()

    def step(self, action):
        """
        Takes an action in the environment and updates the agent's position.

        Parameters:
        - action (tuple): The action to be taken.

        Returns:
        - new_state: The new state after taking the action.
        - reward: The reward received after taking the action.
        - done: A boolean indicating if the goal has been reached.
        """
        # Calculate the new position after taking the action
        new_x = self.agent_position[0] + action[0]
        new_y = self.agent_position[1] + action[1]

        # Check if the new position is within the grid bounds and not an obstacle
        if (0 <= new_x < self.width) and (0 <= new_y < self.height) and not (new_x, new_y) in self.obstacles:
            # Update the agent's position
            self.agent_position = (new_x, new_y)

        # Check if the new position is the goal
        done = self.agent_position == self.goal

        # Define the reward for reaching the goal or taking a step
        reward = 0 if done else -1

        return self.get_state(), reward, done

    def reset(self):
        """
        Resets the grid and agent position to the initial state.
        """
        self.agent_position = self.start
        self.grid = np.zeros((self.height, self.width))
        for obstacle in self.obstacles:
            self.grid[obstacle] = -1
        self.grid[self.goal] = 1
        return self.get_state()  # Return the initial state

    def render(self):
        """
        Renders the current state of the gridworld.
        """
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) == self.agent_position:
                    print('A', end=' ')  # Agent's position
                elif (x, y) == self.goal:
                    print('G', end=' ')  # Goal position
                elif (x, y) in self.obstacles:
                    print('#', end=' ')  # Obstacle
                else:
                    print('.', end=' ')  # Empty cell
            print()  # Newline at the end of each row

        print()  # Extra newline for better separation between steps

# Example usage
gridworld = Gridworld(width=5, height=5, start=(0, 0), goal=(4, 4), obstacles=[(1, 1), (2, 2), (3, 3)])
gridworld.render()
```

### Summary
This section established the basics of RL and introduced a simple Gridworld environment for future exploration.

## Part 2: Building the Agent

### Implementing the Agent
We now add an agent to interact with the Gridworld, beginning with a basic agent that makes random moves.

#### Agent and Policy
- **Agent**: The entity that acts in the environment.
- **Policy**: The decision-making strategy of the agent.

#### The RandomAgent
We implement a `RandomAgent` class, which serves as our initial, naive agent.

- File: `gridworld_agent.py`

```python
# gridworld_agent.py
import random

class RandomAgent:
    def __init__(self, actions):
        """
        Initializes a RandomAgent object.
        
        Parameters:
        - actions (list): A list of possible actions the agent can take.
        """
        self.actions = actions

    def choose_action(self, state):
        """
        Chooses the next action at random from the list of possible actions.
        
        Parameters:
        - state: The current state of the agent (not used in this random policy).
        
        Returns:
        A randomly selected action from the agent's list of possible actions.
        """
        return random.choice(self.actions)
```

#### Agent Actions in Gridworld
The agent can move in four directions: up, down, left, and right.

#### Interactions and Outcomes
We simulate the agent's interactions with the Gridworld, using a reward system to guide its learning.

#### Running the Agent
We execute the agent within the Gridworld, observing its behavior over time.

### Summary and Next Steps
We introduced a simple agent to our Gridworld project. This sets the foundation for more advanced learning algorithms to come.

---

# Full Chapter 1: The Complete Code

We combine the elements from Part 1 and Part 2 to run our Gridworld simulation with the RandomAgent. The code for the agent's interaction with the Gridworld is in the `gridworld_random_agent.py` file.

- File: `gridworld_random_agent.py`

```python
# gridworld_random_agent.py
from gridworld import Gridworld
from gridworld_agent import RandomAgent

# Define the Gridworld environment
gridworld = Gridworld(width=5, height=5, start=(0, 0), goal=(4, 4), obstacles=[(1, 1), (2, 2), (3, 3)])

# Define the actions and create the RandomAgent
actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Actions corresponding to [Up, Right, Down, Left]
agent = RandomAgent(actions)

total_reward = 0

# Run the agent in the environment
for _ in range(100):  # Run for a certain number of steps or until the goal is reached
    current_state = gridworld.agent_position
    action = agent.choose_action(current_state)
    new_state, reward, done = gridworld.step(action)
    total_reward += reward
    print(f"Action taken: {action}")
    print(f"Reward received: {reward}")
    print(f"Total reward: {total_reward}")
    gridworld.render()
    if done:
        print("Goal reached!")
        break
```

### Why These Actions?
We selected standard grid movement actions for simplicity and to establish a baseline for agent behavior.

### Following Equations and Logic
At this stage, the agent's decisions are random. We'll later introduce more sophisticated strategies based on RL algorithms and mathematical foundations.

### Summary
Chapter 1 sets the stage for an exploration into reinforcement learning, providing the fundamental concepts and a practical project to apply these ideas. As we progress, the agents will evolve from making random moves to employing advanced strategies informed by their interactions with the environment. 