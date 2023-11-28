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