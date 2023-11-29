# Chapter 4: Time-Based Reminder Environment

## Overview

In this chapter, we will explore the creation of a time-based reminder environment. The main goal of this environment is to simulate an agent that reminds a human user to complete specific tasks at designated times. We'll be using reinforcement learning principles to train an agent that can learn to provide reminders at the most appropriate moments, taking into account the user's responsiveness and the importance of each reminder.

## Objectives

- Design a reinforcement learning environment that simulates user interactions with a reminder system.
- Develop an agent that can learn to provide timely reminders.
- Implement a framework for evaluating the performance of the reminder agent.

## Introduction to the Time-Based Reminder Environment

A time-based reminder environment is a dynamic system where an agent must learn to interact with a human user by sending reminders at optimal times. The agent will receive feedback based on the user's reactions, which will guide the learning process.

### Key Components of the Reminder Environment

- **Agent**: The reminder system.
- **User**: Simulated user who receives and responds to reminders.
- **State**: The current state includes the time, the list of pending reminders, and the user's status.
- **Action**: Sending a reminder or waiting.
- **Reward**: Feedback representing the effectiveness of the reminder timing.

## Designing the Environment

The reminder environment will be a simulated world where each time step represents a moment in a day. The agent will decide whether to send a reminder at each time step.

### Implementing the Environment

We will create a Python class for our environment, which will feature methods to step through time, send reminders, and receive user feedback.

#### File: `reminder_environment.py`

```python
# reminder_environment.py
import numpy as np

class ReminderEnvironment:
    def __init__(self, reminder_list, user_responsiveness):
        """
        Initializes the ReminderEnvironment object.

        Parameters:
        - reminder_list (list): A list of tuples representing reminders and their deadlines.
        - user_responsiveness (float): A parameter controlling the simulated user's likelihood of responding to reminders.
        """
        self.reminder_list = reminder_list
        self.user_responsiveness = user_responsiveness
        self.current_time = 0
        self.done = False

    def step(self, action):
        """
        Takes an action (send a reminder or wait) and updates the environment's state.

        Parameters:
        - action (int): 0 for wait, 1 for sending a reminder.

        Returns:
        - new_state: The new state after taking the action.
        - reward: The reward received after taking the action.
        - done: A boolean indicating if all reminders are handled.
        """
        reward = 0

        # Simulate user response to a reminder based on user_responsiveness
        if action == 1:  # Send reminder
            reminder, deadline = self.reminder_list[self.current_time]
            user_response = np.random.rand() < self.user_responsiveness
            if user_response and self.current_time <= deadline:
                reward = 10  # Positive reward for successful reminder
            else:
                reward = -5  # Negative reward for ineffective reminder or missed deadline

        # Move to the next time step
        self.current_time += 1
        if self.current_time >= len(self.reminder_list) or all(r[1] < self.current_time for r in self.reminder_list):
            self.done = True

        return self.get_state(), reward, self.done

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        self.current_time = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        """
        Returns the current state of the environment.
        """
        # State can include current time, reminders left, time till next deadline, etc.
        reminders_remaining = len([r for r in self.reminder_list if r[1] >= self.current_time])
        time_till_next_deadline = min([r[1] for r in self.reminder_list if r[1] >= self.current_time], default=0) - self.current_time
        return self.current_time, reminders_remaining, time_till_next_deadline

    def render(self):
        """
        Optional method to render the environment's state to the console.
        """
        print(f"Current time: {self.current_time}")
        print(f"Reminders remaining: {self.get_state()[1]}")
        print(f"Time till next deadline: {self.get_state()[2]}")

# Example usage
if __name__ == "__main__":
    reminders = [(1, 5), (2, 7), (4, 10)]  # Each tuple is (reminder, deadline)
    env = ReminderEnvironment(reminder_list=reminders, user_responsiveness=0.8)
    env.render()
    for _ in range(10):
        action = np.random.choice([0, 1])  # Choose random action (0: wait, 1: send reminder)
        state, reward, done = env.step(action)
        print(f"Action taken: {'Send reminder' if action == 1 else 'Wait'}")
        print(f"Reward received: {reward}")
        env.render()
        if done:
            print("All reminders handled.")
            break
```

## Summary and Next Steps

We have defined a new time-based reminder environment and outlined how to implement its core functionalities. In the upcoming chapters, we will develop an agent that interacts with this environment and learns the best strategy to provide reminders through reinforcement learning. We will explore various RL algorithms and evaluate their effectiveness in achieving the goal of timely and effective reminders.