# gridworld.py
import numpy as np

class Gridworld:
    def __init__(self, width, height, start, goal, obstacles):
        """
        Initializes a Gridworld object.
        """
        self.width = width  # Set the width of the grid
        self.height = height  # Set the height of the grid
        self.start = start  # Set the starting position of the agent
        self.goal = goal  # Set the goal position
        self.obstacles = obstacles  # Set the obstacle positions
        self.agent_position = start  # Initialize the agent's position
        self.grid = np.zeros((self.height, self.width))  # Initialize the grid as a 2D numpy array
        self.reset()
    
    def get_state(self):
        """
        Returns a unique state index for the agent's current position.

        The method converts the 2D coordinates (x, y) of the agent's position 
        into a single number that uniquely identifies each possible state in the grid.
        This is done by using the formula 'y * self.width + x', which maps the 2D 
        coordinates to a unique 1D index. This representation is useful in algorithms 
        that require state representation as a single number, like many reinforcement 
        learning algorithms.
        """
        x, y = self.agent_position
        return y * self.width + x


    def step(self, action):
        """
        Takes an action in the environment and updates the agent's position.
        """
        # Calculate potential new position after the action
        new_x = self.agent_position[0] + action[0]
        new_y = self.agent_position[1] + action[1]

        # Check if the new position is within the grid boundaries and not an obstacle
        # The new position must be within the range of 0 and the grid's width for the x-coordinate,
        # and within the range of 0 and the grid's height for the y-coordinate.
        # Additionally, the new position should not be one of the predefined obstacles.
        if (0 <= new_x < self.width) and (0 <= new_y < self.height) and not (new_x, new_y) in self.obstacles:
            self.agent_position = (new_x, new_y)  # Update agent's position

        # Check if the agent has reached the goal
        done = self.agent_position == self.goal
        # Reward logic: 0 if goal is reached, otherwise -1. Negative rewards are used to 
        # penalize certain actions or states, encouraging the agent to reach the goal efficiently.
        reward = 0 if done else -1

        return self.get_state(), reward, done

    def reset(self):
        """
        Resets the grid and agent position to the initial state.
        """
        self.agent_position = self.start  # Reset agent to the start position
        self.grid = np.zeros((self.height, self.width))  # Reset the grid
        for obstacle in self.obstacles:
            self.grid[obstacle] = -1  # Mark obstacles in the grid
        self.grid[self.goal] = 1  # Mark the goal in the grid
        return self.get_state()

    def render(self):
        """
        Renders the current state of the gridworld.
        """
        for y in range(self.height):
            for x in range(self.width):
                # Print symbols for agent, goal, obstacles, or empty space
                if (x, y) == self.agent_position:
                    print('A', end=' ')  # Agent's current position
                elif (x, y) == self.goal:
                    print('G', end=' ')  # Goal position
                elif (x, y) in self.obstacles:
                    print('#', end=' ')  # Obstacle position
                else:
                    print('.', end=' ')  # Empty cell
            print()  # New line after each row
        print()  # Additional new line for separation