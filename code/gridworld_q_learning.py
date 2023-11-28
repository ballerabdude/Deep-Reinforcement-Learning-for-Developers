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

        # print(f"Episode {episode + 1}: Next state: {next_state}, Reward: {reward}, Done: {done}")

        # Update the Q-table with the action index
        agent.update_q_table(current_state, action_index, reward, next_state)

        # Update total reward and current state
        total_reward += reward
        current_state = next_state

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")
