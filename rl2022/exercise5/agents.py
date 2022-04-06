from abc import ABC, abstractmethod
from collections import defaultdict
import random
from typing import List, Dict, DefaultDict
import numpy as np
from gym.spaces import Space
from gym.spaces.utils import flatdim


class MultiAgent(ABC):
    """Base class for multi-agent reinforcement learning

    **DO NOT CHANGE THIS BASE CLASS**

    """

    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of MARL agents
        namely epsilon, learning rate and discount rate.

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)

        :attr n_acts (List[int]): number of actions for each agent
        """

        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma

    @abstractmethod
    def act(self) -> List[int]:
        """Chooses an action for all agents for stateless task

        :return (List[int]): index of selected action for each agent
        """
        ...

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


class IndependentQLearningAgents(MultiAgent):
    """Agent using the Independent Q-Learning algorithm

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of IndependentQLearningAgents

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping actions ACTs
            to respective Q-values for all agents

        Initializes some variables of the Independent Q-Learning agents, namely the epsilon, discount rate
        and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for i in range(self.num_agents)]


    def act(self) -> List[int]:
        """Implement the epsilon-greedy action selection here for stateless task

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :return (List[int]): index of selected action for each agent
        """
        actions = []
        ### PUT YOUR CODE HERE ###
        for i in range(self.num_agents):
            if np.random.random() < self.epsilon or len(self.q_tables[i])==0:
                act = np.random.randint(0,self.n_acts,1,dtype=int)
                # print(act)
                actions.append(act[0])
            else:
                q_table = self.q_tables[i]
                actions.append(max(q_table, key=q_table.get))
        return actions

    def learn(
        self, actions: List[int], rewards: List[float], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current actions of each agent
        """
        updated_values = []
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q5")
        Q=[]
        for i in range(self.num_agents):
            q_table = self.q_tables[i]
            q = q_table[actions[i]]
            max_q = 0
            
            for a_index in range(int(self.n_acts[0])):
                Q.append(q_table[a_index])
            max_q = max(Q) if not dones[i] else 0
            q_table[actions[i]]=q + self.learning_rate*(rewards[i] + self.gamma*max_q - q)
            #q_table[i] = q + self.learning_rate*(rewards[i] - q)
            self.q_tables[i] = q_table
            updated_values.append(q_table)
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        max_deduct, decay = 0.95, 0.07
        #self.learning_rate = 1.0 - (min(1.0, timestep / (decay * max_timestep))) * max_deduct
        self.epsilon = 1.0 - (min(1.0, timestep / (decay * max_timestep))) * max_deduct
        #self.learning_rate = max(0.07-0.007*timestep/max_timestep,0.01)
        self.gamma = 0.99
class JointActionLearning(MultiAgent):
    """
    Agents using the Joint Action Learning algorithm with Opponent Modelling

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of JointActionLearning

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping joint actions ACTs
            to respective Q-values for all agents
        :attr models (List[DefaultDict]): each agent holding model of other agent
            mapping other agent actions to their counts

        Initializes some variables of the Joint Action Learning agents, namely the epsilon, discount rate and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_acts = [flatdim(action_space) for action_space in self.action_spaces]

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for _ in range(self.num_agents)]

        # initialise models for each agent mapping state to other agent actions to count of other agent action
        # in state
        self.models = [defaultdict(lambda: 0) for _ in range(self.num_agents)] 
    def act(self) -> List[int]:
        """Implement the epsilon-greedy action selection here for stateless task

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :return (List[int]): index of selected action for each agent
        """
        joint_action = []
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q5")
        if self.num_agents == 0:
            ev = 0                
        for i in range(self.num_agents):
            q_table = self.q_tables[i]
            ev = 1
            if random.random() < self.epsilon or ev == 0:
                joint_action.append(np.random.randint(0,self.n_acts,1,dtype=int)[0])
            else:
                for self_action in range(self.action_spaces[0].n):
                    exp_vals = 0
                    evs = []
                    for others_action in range(self.action_spaces[0].n):
                        for n_others_action in range(self.action_spaces[0].n):
                            if self.models[i][n_others_action]==0:
                                exp_vals=0
                            else:
                                action_comb = (self_action, others_action)
                                exp_vals += (self.models[i][others_action] / self.models[i][n_others_action]) * q_table[action_comb]
                    evs.append(exp_vals)
                ev_max = max(evs)
                acts = [index for index, exp in enumerate(evs) if exp == ev_max]
                joint_action.append(random.choice(acts))
        return joint_action

    def learn(
        self, actions: List[int], rewards: List[float], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables and models based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        updated_values = []
        ### PUT YOUR CODE HERE ###
        for i in range(self.num_agents):
            self_action = actions[i]
            others_action = actions[1-i]
            q_table = self.q_tables[i]
            self.models[i][others_action]+=1 if self.models[i][others_action] else 1
            q_val = q_table[(self_action,others_action)]
            evs=[]
            for n_self_action in range(self.n_acts[i]):
                ev=0
                for n_others_action in range(self.n_acts[i]):
                    if self.models[i][n_others_action] == 0:
                        ev = 0
                    else:
                        action_comb = (n_self_action, n_others_action)
                        ev+=(self.models[i][others_action]/self.models[i][n_others_action]) * q_table[action_comb]
                evs.append(ev)
            ev_max = max(evs) if not dones[i] else 0
            q_table[tuple(actions)] = q_val + self.learning_rate*(rewards[i] + self.gamma * ev_max - q_val)
            self.q_tables[i]=q_table
            updated_values.append(q_table[tuple(actions)])
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        max_deduct, decay = 0.95, 0.07
        #self.learning_rate = 1.0 - (min(1.0, timestep / (decay * max_timestep))) * max_deduct
        self.epsilon = 1.0 - (min(1.0, timestep / (decay * max_timestep))) * max_deduct
        #self.learning_rate = max(0.005-0.0005*timestep/max_timestep,0.001)
        self.gamma = 0.99# - (min(0.8, timestep / (decay * max_timestep))) * max_deduct
