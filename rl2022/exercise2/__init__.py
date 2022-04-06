from abc import ABC, abstractmethod
from collections import defaultdict
import random
from typing import List, Dict, DefaultDict
from gym.spaces import Space
from gym.spaces.utils import flatdim
import numpy as np

class Agent(ABC):
    """Base class for Q-Learning agent

    **ONLY CHANGE THE BODY OF THE act() FUNCTION**

    """

    def __init__(
        self,
        action_space: Space,
        obs_space: Space,
        gamma: float,
        epsilon: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of the Q-Learning agent
        namely the epsilon, learning rate and discount rate.

        :param action_space (int): action space of the environment
        :param obs_space (int): observation space of the environment
        :param gamma (float): discount factor (gamma)
        :param epsilon (float): epsilon for epsilon-greedy action selection

        :attr n_acts (int): number of actions
        :attr q_table (DefaultDict): table for Q-values mapping (OBS, ACT) pairs of observations
            and actions to respective Q-values
        """

        self.action_space = action_space
        self.obs_space = obs_space
        self.n_acts = flatdim(action_space)

        self.epsilon: float = epsilon
        self.gamma: float = gamma

        self.q_table: DefaultDict = defaultdict(lambda: 0)

    def act(self, obs: int) -> int:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obs (int): received observation representing the current environmental state
        :return (int): index of selected action
        """
        ### PUT YOUR CODE HERE ###

        best_action=0
        Q=np.zeros([1,self.n_acts])
        for a_index in range(self.n_acts):
            Q[0,a_index]=self.q_table[(obs,a_index)]
        best_action=np.argmax(Q[0])
        ### RETURN AN ACTION HERE ###
        if random.random() >= self.epsilon:
            return best_action
        else:
            return self.action_space.sample()

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def learn(self):
        ...


class QLearningAgent(Agent):
    """
    Agent using the Q-Learning algorithm

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, alpha: float, **kwargs):
        """Constructor of QLearningAgent

        Initializes some variables of the Q-Learning agent, namely the epsilon, discount rate
        and learning rate alpha.

        :param alpha (float): learning rate alpha for Q-learning updates
        """

        super().__init__(**kwargs)
        self.alpha: float = alpha

    def learn(
        self, obs: int, action: int, reward: float, n_obs: int, done: bool
    ) -> float:
        """Updates the Q-table based on agent experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obs (int): received observation representing the current environmental state
        :param action (int): index of applied action
        :param reward (float): received reward
        :param n_obs (int): received observation representing the next environmental state
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (float): updated Q-value for current observation-action pair
        """
        ### PUT YOUR CODE HERE ###
        q = self.q_table[(obs,action)]
        max_q = 0
        Q=np.zeros([1,self.n_acts])
        for a_index in range(self.n_acts):
            Q[0,a_index]=self.q_table[(n_obs,a_index)]
        max_q = max(Q[0])
        self.q_table[(obs,action)]=float(q + self.alpha*(reward + self.gamma*max_q - q))

        return self.q_table[(obs, action)]

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        self.epsilon-=self.epsilon/max_timestep
        #raise NotImplementedError("Needed for Q2")


class MonteCarloAgent(Agent):
    """
    Agent using the Monte-Carlo algorithm for training

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, **kwargs):
        """Constructor of MonteCarloAgent

        Initializes some variables of the Monte-Carlo agent, namely epsilon,
        discount rate and an empty observation-action pair dictionary.

        :attr sa_counts (Dict[(Obs, Act), int]): dictionary to count occurrences observation-action pairs
        """
        super().__init__(**kwargs)
        self.sa_counts = {}

    def learn(
        self, obses: List[int], actions: List[int], rewards: List[float]
    ) -> Dict:
        """Updates the Q-table based on agent experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obses (List(int)): list of received observations representing environmental states
            of trajectory (in the order they were encountered)
        :param actions (List[int]): list of indices of applied actions in trajectory (in the
            order they were applied)
        :param rewards (List[float]): list of received rewards during trajectory (in the order
            they were received)
        :return (Dict): A dictionary containing the updated Q-value of all the updated state-action pairs
            indexed by the state action pair.
        """
        updated_values = {}
        ### PUT YOUR CODE HERE ###
        G: DefaultDict = defaultdict(lambda: 0)
        for i in reversed(range(len(obses)-1)):
            G[(obses[i],actions[i])] = rewards[i+1]+ self.gamma*G[(obses[i],actions[i])]
        for i in range(len(obses)):
            if (obses[i],actions[i]) not in self.sa_counts.keys():
                self.sa_counts[(obses[i],actions[i])]=1
            else:
                self.sa_counts[(obses[i],actions[i])]+=1
            if (obses[i],actions[i]) not in list(updated_values.keys())[:i]:
                updated_values[(obses[i],actions[i])]=G[(obses[i],actions[i])]
                #updated_values[(obses[i],actions[i])]=G[(obses[i],actions[i])]]/self.sa_counts[(obses[i],actions[i])]
                self.q_table[(obses[i],actions[i])] = (self.q_table[(obses[i],actions[i])]*(self.sa_counts[(obses[i],actions[i])] - 1) + G[(obses[i],actions[i])])/self.sa_counts[(obses[i],actions[i])]
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        max_deduct, decay = 0.95, 0.07
        self.epsilon = 1.0 - (min(1.0, timestep / (decay * max_timestep))) * max_deduct