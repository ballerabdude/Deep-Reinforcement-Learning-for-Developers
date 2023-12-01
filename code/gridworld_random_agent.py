from gridworld import Gridworld
from gridworld_agent import RandomAgent

# Define the Gridworld environment
gridworld = Gridworld(width=5, height=5, start=(0, 0), goal=(4, 4), obstacles=[(1, 1), (2, 2), (3, 3)])

# Define the actions and create the RandomAgent
actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Actions corresponding to [Up, Right, Down, Left]
agent = RandomAgent(actions)

total_reward = 0

# Run the agent in the environment
episodes = 100
for _ in range(episodes):  # Run for a certain number of steps or until the goal is reached
    current_state = gridworld.agent_position
    action = agent.choose_action(current_state)
    new_state, reward, done = gridworld.step(action)
    total_reward += reward
    if done:
        print("Goal reached!")
        print(f"Total reward: {total_reward}")
        break
    # if at the last step, and not done, print the total reward
    if _ == episodes - 1 and not done:
        print("Goal not reached!")
        print(f"Total reward: {total_reward}")