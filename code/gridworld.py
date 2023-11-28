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
    def get_state(self):
        """
        Returns a unique state index for the agent's current position.
        """
        x, y = self.agent_position
        return y * self.width + x

# Example usage
gridworld = Gridworld(width=5, height=5, start=(0, 0), goal=(4, 4), obstacles=[(1, 1), (2, 2), (3, 3)])
gridworld.render()