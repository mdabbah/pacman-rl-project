import random, util

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical

from game import Agent
from game import Actions
from experiments import timer

#     ********* Reflex agent- sections a and b *********
from pacman import GameState


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """
    def __init__(self):
        self.lastPositions = []
        self.dc = None


    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
        ------------------------------------------------------------------------------
        """
        # Collect legal moves and successor states
        with timer():
            legalMoves = gameState.getLegalActions()

            # Choose one of the best actions
            scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
            bestScore = max(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best


        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current GameState (pacman.py) and the proposed action
        and returns a number, where higher numbers are better.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return betterEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.
    """
    return gameState.getScore()

######################################################################################
# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
    """

    The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

    A GameState specifies the full game state, including the food, capsules, agent configurations and more.
    Following are a few of the helper methods that you can use to query a GameState object to gather information about
    the present state of Pac-Man, the ghosts and the maze:

    gameState.getLegalActions():
    gameState.getPacmanState():
    gameState.getGhostStates():
    gameState.getNumAgents():
    gameState.getScore():
    The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
    """

    state_eval = gameState.getScore()

    # get my position
    my_pos = gameState.getPacmanPosition()
    # gt = gameState.construct_state_tensor()
    manhattan_dists_from_ghosts = [util.manhattanDistance(my_pos, pos) for pos in gameState.getGhostPositions()]
    manhattan_dists_from_food = [util.manhattanDistance(my_pos, pos) for pos in getFoodPositions(gameState)]
    manhattan_dists_from_capsuls = [util.manhattanDistance(my_pos, pos) for pos in gameState.getCapsules()]

    ghosts_scared = [ghost_state.scaredTimer > 0 for ghost_state in gameState.getGhostStates()]
    # [print(ghost_state.scaredTimer > 0) for ghost_state in gameState.getGhostStates()]
    amplification_factor_ghost = 1
    # if all(ghosts_scared):
    #     amplification_factor_ghost *= -1  # if ghosts are scared we prefer to be near them

    if manhattan_dists_from_ghosts:  # only if list is not empty
        distance_from_nearest_ghost = min(min(manhattan_dists_from_ghosts), 5)
        state_eval += amplification_factor_ghost*distance_from_nearest_ghost
        # the farther from the nearest ghost the better

    amplification_factor_food = 1
    if manhattan_dists_from_food:  # only if list is not empty
        distance_from_nearest_food = np.mean(manhattan_dists_from_food)
        state_eval -= amplification_factor_food*distance_from_nearest_food
        # the farther from the nearest food the worse

    amplification_factor_capsule = 1
    if manhattan_dists_from_capsuls:  # only if list is not empty
        distance_from_nearest_capsule = min(min(manhattan_dists_from_capsuls), 5)
        state_eval -= amplification_factor_capsule*distance_from_nearest_capsule
        # the farther from the nearest food the worse

    return state_eval  # gameState.getScore()


def getFoodPositions(gameState):
    """
    returns a list [of food capsules positions(tuple(x,y) on the grid]
    :param gameState: game state
    :return: list[tuple(x,y)]
    """

    food_poss = []
    food_map = gameState.getFood()
    for y in range(food_map.height):
        for x in range(food_map.width):

            if food_map[x][y]:
                food_poss.append((x, y))

    return food_poss


#     ********* MultiAgent Search Agents- sections c,d,e,f*********

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

######################################################################################
# c: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction. Terminal states can be found by one of the following:
          pacman won, pacman lost or there are no legal moves.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.getScore():
            Returns the score corresponding to the current state of the game

          gameState.isWin():
            Returns True if it's a winning state

          gameState.isLose():
            Returns True if it's a losing state

          self.depth:
            The depth to which search should continue

        """

        with timer():
            actions = gameState.getLegalActions(self.index)
            next_states = [gameState.generateSuccessor(0, action) for action in actions]
            minimax_scores = [self.getMiniMaxScores(state, self.depth, self.index) for state
                                in next_states]

            bestScore = max(minimax_scores)
            bestIndices = [index for index in range(len(minimax_scores)) if minimax_scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        # gameState.
        return actions[chosenIndex]

    def getMiniMaxScores(self, gameState, depth, agent_index):
        """
        returns minimax score recursively
        :param depth:
        :param agent_index:
        :return:
        """
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()

        if depth == 0:
            return self.evaluationFunction(gameState)

        # turn update
        next_agent_index = agent_index + 1
        next_agent_index = next_agent_index % gameState.getNumAgents()
        if next_agent_index == 0:
            depth -= 1

        # child generation
        actions = gameState.getLegalActions(next_agent_index)
        next_states = [gameState.generateSuccessor(next_agent_index, action) for action in actions]

        minimax_scores = [self.getMiniMaxScores(next_state, depth, next_agent_index)
                          for next_state in next_states]

        if next_agent_index == 0:
            # print("max")
            return max(minimax_scores)
        else:
            # print("mini")
            return min(minimax_scores)



######################################################################################
# d: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        with timer():
            actions = gameState.getLegalActions(self.index)
            next_states = [gameState.generateSuccessor(0, action) for action in actions]
            AlphaBeta_scores = [self.AlphaBetaAction(self.depth, self.index, state, -np.inf, np.inf) for state
                                in next_states]

            best_score = max(AlphaBeta_scores)
            top_index = [index for index in range(len(AlphaBeta_scores)) if AlphaBeta_scores[index] == best_score]
            chosen = random.choice(top_index)
        return actions[chosen]

    def AlphaBetaAction(self, depth, agent_index, gameState, Alpha, Beta):
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()
        if depth == 0:
            return self.evaluationFunction(gameState)

        # turn update
        next_agent_index = agent_index + 1
        next_agent_index = next_agent_index % gameState.getNumAgents()
        if next_agent_index == 0:
            depth -= 1

        # child generation
        actions = gameState.getLegalActions(next_agent_index)
        next_states = [gameState.generateSuccessor(next_agent_index, action) for action in actions]

        if next_agent_index == 0:
            cur_max = -np.inf
            for state in next_states:
                v = self.AlphaBetaAction(depth, next_agent_index, state, Alpha, Beta)
                cur_max = max(cur_max, v)
                Alpha = max(cur_max, Alpha)
                if cur_max >= Beta:
                    return np.inf
            return cur_max
        else:
            cur_min = np.inf
            for state in next_states:
                v = self.AlphaBetaAction(depth, next_agent_index, state, Alpha, Beta)
                cur_min = min(cur_min, v)
                Beta = min(cur_min, Beta)
                if cur_min <= Alpha:
                    return -np.inf
            return cur_min


######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their legal moves.
        """
        with timer():
            actions = gameState.getLegalActions(self.index)
            next_states = [gameState.generateSuccessor(self.index, action) for action in actions]
            expectimax_scores = [self.getExpectiMaxScores(state, self.depth, self.index) for state
                              in next_states]

            bestScore = max(expectimax_scores)
            bestIndices = [index for index in range(len(expectimax_scores)) if expectimax_scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return actions[chosenIndex]

    def getExpectiMaxScores(self, gameState, depth, agent_index):
        """
        returns the expectimax value to the given state for the given depth
        for agent index
        :param state:
        :param depth:
        :param agent_index:
        :return:
        """
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()

        if depth == 0:
            return self.evaluationFunction(gameState)

        # turn update
        next_agent_index = agent_index + 1
        next_agent_index = next_agent_index % gameState.getNumAgents()
        if next_agent_index == 0:
            depth -= 1

        # child generation
        actions = gameState.getLegalActions(next_agent_index)
        next_states = [gameState.generateSuccessor(next_agent_index, action) for action in actions]

        minimax_scores = [self.getExpectiMaxScores(next_state, depth, next_agent_index)
                          for next_state in next_states]

        if next_agent_index == 0:
            # print("max")
            return max(minimax_scores)
        else:

            return np.mean(minimax_scores)


######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
        """
        with timer():
            actions = gameState.getLegalActions(self.index)
            next_states = [gameState.generateSuccessor(self.index, action) for action in actions]
            expectimax_scores = [self.getExpectiMaxScores(state, self.depth, self.index) for state
                                 in next_states]

            bestScore = max(expectimax_scores)
            bestIndices = [index for index in range(len(expectimax_scores)) if expectimax_scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
            # gameState.getGhostState(3).
        return actions[chosenIndex]


    def getExpectiMaxScores(self, gameState, depth, agent_index):
        """
        returns the expectimax value to the given state for the given depth
        for agent index
        :param state:
        :param depth:
        :param agent_index:
        :return:
        """
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()

        if depth == 0:
            return self.evaluationFunction(gameState)

        # turn update
        next_agent_index = agent_index + 1
        next_agent_index = next_agent_index % gameState.getNumAgents()
        if next_agent_index == 0:
            depth -= 1

        # child generation
        actions = gameState.getLegalActions(next_agent_index)
        next_states = [gameState.generateSuccessor(next_agent_index, action) for action in actions]

        if next_agent_index == 0:
            minimax_scores = [self.getExpectiMaxScores(next_state, depth, next_agent_index)
                              for next_state in next_states]
            return max(minimax_scores)
        else:
            dist = self.getGhostDisturbution(gameState, next_agent_index)
            next_states_probs = [dist[action] for action in actions]

            minimax_scores = [self.getExpectiMaxScores(next_state, depth, next_agent_index) * next_states_probs[idx]
                              for idx, next_state in enumerate(next_states)]

            return np.sum(minimax_scores)

    @classmethod
    def getGhostDisturbution(cls, state, index):
        """
        given a state and a ghost index
        :param gameState:
        :return:
        """
        # Read variables from state
        ghostState = state.getGhostState(index)
        legalActions = state.getLegalActions(index)
        pos = state.getGhostPosition(index)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5

        actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
        newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [util.manhattanDistance(pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = 0.8
        else:
            bestScore = min(distancesToPacman)
            bestProb = 0.8
        bestActions = [action for action, distance in zip(legalActions, distancesToPacman) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions: dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += (1 - bestProb) / len(legalActions)
        dist.normalize()
        return dist

######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
    """
      Your competition agent
    """

    def getAction(self, gameState):
        """
          Returns the action using self.depth and self.evaluationFunction

        """

        # BEGIN_YOUR_CODE
        raise Exception("Not implemented yet")
        # END_YOUR_CODE


class OptimalAgent(nn.Module, MultiAgentSearchAgent):
    """
      Your competition agent
    """

    def __init__(self, state_size=840, number_of_actions=4, HL1_size=1024, HL2_size=500,
                 gamma=0.99, explor_factor=0.1, clap_grads=True, R=None, device=None):
        super(OptimalAgent, self).__init__()
        self.index = 0  # by convention pacman is index 0
        self.input_layer = nn.Linear(state_size, HL1_size)
        self.hidden_layer1 = nn.Linear(HL1_size, HL2_size)
        self.action_output = nn.Linear(HL2_size, number_of_actions)
        self.value_output = nn.Linear(HL2_size, 1)

        self.device = device

        self.is_trainable = True
        self.should_record = True
        self.rewards = []
        self.action_log_probs = []
        self.state_values = []
        self.actual_rewards = []

        self.gamma = gamma
        self.optimizer = None
        self.clip_grads = clap_grads
        self.reward_approximator = R
        self.prev_score = 0

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
            # self.actual_rewards.append(gameState.getScore() - self.prev_score)
            self.actual_rewards.append(calc_approximate_reward(gameState))
            self.prev_score = gameState.getScore()
            self.action_log_probs.append(pi_s.log_prob(chosen_action_idx))
            self.state_values.append(value)
            self.rewards.append(self.reward_approximator(state_vector))

        chosen_action = all_actions[chosen_action_idx]
        return chosen_action

    def set_optimizer(self, optimizer):
        """

        :param optimizer: optimerzer object from the torch.optim library
        :return: None
        """
        if not isinstance(optimizer,torch.optim.Optimizer):
            raise ValueError(' the given optimizer is not supported'
                             'please provide an optimizer that is an instance of'
                             'torch.optim.Optimizer')
        self.optimizer = optimizer

    def set_trainable(self, is_trainable: bool)-> None:
        # if self.is_trainable == is_trainable:
        #     return

        self.is_trainable = is_trainable
        for p in self.parameters():
            p.requires_grad = is_trainable

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
        del self.actual_rewards[:]
        return float(total_loss)

    def get_cumulative_return(self):
        rewards = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        rewards = torch.stack(rewards).to(self.device)
        return rewards.sum()

    def reset_saved_stats(self):
        del self.rewards[:]
        del self.action_log_probs[:]
        del self.state_values[:]
        del self.actual_rewards[:]


def calc_approximate_reward(game_state: GameState):


    state_tensor = game_state.construct_state_tensor()

    # get my position
    my_pos = game_state.getPacmanPosition()

    # gt = gameState.construct_state_tensor()
    manhattan_dists_from_ghosts = [util.manhattanDistance(my_pos, pos) for pos in game_state.getGhostPositions()]

    num_foods = np.sum(state_tensor[0])
    num_capsules = np.sum(state_tensor[4])
    ghosts_scared = [ghost_state.scaredTimer > 0 for ghost_state in game_state.getGhostStates()]

    if manhattan_dists_from_ghosts:
        min_dist_from_ghost = min(manhattan_dists_from_ghosts)
        min_dist_from_ghost = -min_dist_from_ghost if ghosts_scared else min_dist_from_ghost

        return -1*num_foods*10 -1*num_capsules*30 + min_dist_from_ghost

    return -1*num_foods*10 -1*num_capsules*30