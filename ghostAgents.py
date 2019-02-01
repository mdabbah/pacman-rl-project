import torch

from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class GhostAgent( Agent ):
  def __init__( self, index ):
    self.index = index

  def getAction( self, state ):
    dist = self.getDistribution(state)
    if len(dist) == 0: 
      return Directions.STOP
    else:
      return util.chooseFromDistribution( dist )
    
  def getDistribution(self, state):
    "Returns a Counter encoding a distribution over actions from the provided state."
    util.raiseNotDefined()

class RandomGhost( GhostAgent ):
  "A ghost that chooses a legal action uniformly at random."
  def getDistribution( self, state ):
    dist = util.Counter()
    for a in state.getLegalActions( self.index ): dist[a] = 1.0
    dist.normalize()
    return dist

class DirectionalGhost( GhostAgent ):
  "A ghost that prefers to rush Pacman, or flee when scared."
  def __init__( self, index, prob_attack=0.8, prob_scaredFlee=0.8 ):
    self.index = index
    self.prob_attack = prob_attack
    self.prob_scaredFlee = prob_scaredFlee
      
  def getDistribution( self, state ):
    # Read variables from state
    ghostState = state.getGhostState( self.index )
    legalActions = state.getLegalActions( self.index )
    pos = state.getGhostPosition( self.index )
    isScared = ghostState.scaredTimer > 0
    
    speed = 1
    if isScared: speed = 0.5
    
    actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
    newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
    pacmanPosition = state.getPacmanPosition()

    # Select best actions given the state
    distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
    if isScared:
      bestScore = max( distancesToPacman )
      bestProb = self.prob_scaredFlee
    else:
      bestScore = min( distancesToPacman )
      bestProb = self.prob_attack
    bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]
    
    # Construct distribution
    dist = util.Counter()
    for a in bestActions: dist[a] = bestProb / len(bestActions)
    for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
    dist.normalize()
    return dist


class OptimalGhost(nn.Module, GhostAgent):
    """
          Your competition agent
        """

    def __init__(self, index, state_size=840, number_of_actions=4, HL1_size=1024, HL2_size=500,
                 gamma=0.99, explor_factor=0.1, clap_grads=True, R=None, device=None):
        super(OptimalGhost, self).__init__()
        self.index = index
        self.input_layer = nn.Linear(state_size, HL1_size)
        self.hidden_layer1 = nn.Linear(HL1_size, HL2_size)
        self.action_output = nn.Linear(HL2_size, number_of_actions)
        self.value_output = nn.Linear(HL2_size, 1)

        self.device = device

        self.rewards = []
        self.action_log_probs = []
        self.state_values = []

        self.gamma = gamma
        self.optimizer = None
        self.clip_grads = clap_grads
        self.reward_approximator = R
        self.is_trainable = True
        self.should_record = True

        # load the neural network or initialize it

    def forward(self, state):
        x = self.input_layer(state)
        x = F.relu(x)
        x = self.hidden_layer1(x)
        x = F.relu(x)

        actions = self.action_output(x)
        value = self.value_output(x)

        return F.softmax(actions, dim=0), value

    def getAction(self, gameState):
        """
          Returns the action using a nural network

        """
        state_vector = gameState.construct_state_tensor().flatten()
        state_vector = torch.tensor(state_vector, device=self.device).float()
        action_dist, value = self.forward(state_vector)

        legal_actions = gameState.getLegalActions(self.index)
        all_actions = ['North', 'South', 'East', 'West']

        pi_s = Categorical(action_dist)
        chosen_action_idx = pi_s.sample()

        if all_actions[chosen_action_idx] not in legal_actions:
            chosen_action_idx = torch.randint(0, len(legal_actions), [1], device=self.device)
            all_actions = legal_actions

        if self.is_trainable or self.should_record:
            self.action_log_probs.append(pi_s.log_prob(chosen_action_idx))
            self.state_values.append(value)
            self.rewards.append(-self.reward_approximator(state_vector))

        chosen_action = all_actions[chosen_action_idx]

        return chosen_action

    def set_optimizer(self, optimizer):
        """

        :param optimizer: optimerzer object from the torch.optim library
        :return: None
        """
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError(' the given optimizer is not supported'
                             'please provide an optimizer that is an instance of'
                             'torch.optim.Optimizer')
        self.optimizer = optimizer

    def set_trainable(self, is_trainable: bool) -> None:
        self.is_trainable = is_trainable
        for p in self.parameters():
            p.requires_grad = is_trainable

    def reset_saved_stats(self):
        del self.rewards[:]
        del self.action_log_probs[:]
        del self.state_values[:]

    def get_cumulative_return(self):
        rewards = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        if rewards == []:
            return torch.tensor([0]).float().to(self.device).squeeze()
        rewards = torch.stack(rewards).to(self.device)
        return rewards.sum()

    def update(self, episode):
        """
        after an episode is finished use this method to update the policy the agent has learned so far acording
        to the monte carlo samples the agent have seen during the episode
        episode parameter is for "episode normalization" code which is not used
        :return: policy loss
        """
        if self.optimizer is None:
            raise ValueError('optimizer not set!'
                             'please use agent.set_optimizer method to set an optimizer')

        R = 0
        policy_loss = []
        value_losses = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        if rewards == []:
            return float(0)
        rewards = torch.stack(rewards).to(self.device)

        eps = np.finfo(np.float32).eps.item()
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for log_prob, reward, v_s in zip(self.action_log_probs, rewards, self.state_values):
            advantage = reward - v_s.item()
            policy_loss.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(v_s.squeeze(), torch.tensor([reward], device=self.device)))

        total_loss = torch.stack(policy_loss).sum().to(self.device) + torch.stack(value_losses).sum().to(self.device)
        self.optimizer.zero_grad()
        total_loss.backward()

        if self.clip_grads:
            for param in self.parameters():
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        del self.rewards[:]
        del self.action_log_probs[:]
        del self.state_values[:]
        return float(total_loss)