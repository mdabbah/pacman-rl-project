import numpy as np
import pickle

from pacman import readCommand, ClassicGameRules
from submission import AlphaBetaAgent, OptimalAgent, MultiAgentSearchAgent, calc_approximate_reward
from ghostAgents import OptimalGhost, GhostAgent, DirectionalGhost
from typing import List, Dict

import torch
from torch import nn
import torch.nn.functional as F
import time, sys
from random import  shuffle

logfile = 'log.txt'
std_out_orig = sys.stdout


class Logger(object):
    def __init__(self, file_name):
        self.terminal = std_out_orig
        self.log = file_name
        self.mute = False

    def log_mute(self):
        self.mute = True

    def log_unmute(self):
        self.mute = False

    def write(self, message):
        self.terminal.write(message)
        if not self.mute:
            with open(self.log, 'a') as f:
                f.write(message)


class RewardApprocimator(nn.Module):

    def __init__(self, input_size=840, HL_size=256, device='gpu', clip_grads=True):

        super(RewardApprocimator, self).__init__()
        self.input_layer = nn.Linear(input_size, HL_size)
        self.hidden_layer1 = nn.Linear(HL_size, 1)

        self.is_trainable = True
        self.rewards = []
        self.device = torch.device("cuda:0" if device == 'gpu' else "cpu")

        self.clip_grads = clip_grads
        self.optimizer = None

    def forward(self, state):
        x = self.input_layer(state)
        x = F.relu(x)
        reward = self.hidden_layer1(x)

        return reward

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


def sample_starting_state(agent_idx=0):

    with open('./rec_games/all_games.list', 'rb') as f:
        all_games = pickle.load(f)

    states_to_poll = [s for s in all_games if s[1] == agent_idx]
    lottery_ticket = np.random.randint(0, len(states_to_poll))

    return states_to_poll[lottery_ticket]


def apply_to_agents(agents, lam):
    for agent in agents:
        lam(agent)


def Nash_GANs_training(Hero_agent: OptimalAgent, Villans: List[OptimalGhost], Kg: int, Kcycle: int,
                       R_model: RewardApprocimator, T=64, is_hero_update=True):
    """
    this is algorithm 2 of the paper with the modified changes
    :param R_model: NN of the rewards
    :param Hero_agent: pacman NN agent
    :param Villans: list of ghosts NN agents
    :param Kg: must be less than Kcycle, represents the cycle duty invested training the villains
    :param Kcycle: the length of the cycle
    :return: estimation of the cumulative reward of pacman over T games
    """
    agent_to_train = 'pacman' if is_hero_update else 'ghosts'
    print("Entering Nash GANs training for " + agent_to_train)
    sys.stdout.log_mute()

    R_model.set_trainable(False)
    Hero_agent.set_trainable(True)
    for g in Villans:
        g.set_trainable(True)

    stopped_improving = False
    i = 0
    num_villains = len(Villans)
    while not stopped_improving:
        if i % Kcycle < Kg:

            villain_to_update = int((i % Kg)/int(Kg/num_villains))
            Hero_agent.set_trainable(False)
            Villans[1-villain_to_update].set_trainable(False)  # assumes only 2 ghosts
            Villans[villain_to_update].set_trainable(True)

            run_game(Hero_agent, Villans, villain_to_update)
            Villans[villain_to_update].update(i)

            apply_to_agents(Villans + [Hero_agent], lambda a: a.reset_saved_stats())
            # train ghosts to kick ass
        else:
            Hero_agent.set_trainable(True)
            for g in Villans:
                g.set_trainable(False)

            run_game(Hero_agent, Villans, starting_agent_idx=0)
            Hero_agent.update(i)
            apply_to_agents(Villans + [Hero_agent], lambda a: a.reset_saved_stats())
            # train pacman to kick ass

        if i > Kcycle:  #*4:
            break

        i += 1

    V_f = []
    V_g = []
    Hero_agent.should_record = True
    for g in Villans:
        g.should_record = True
    for i in range(T):
        run_game(Hero_agent, Villans, starting_agent_idx=0)

        V_f.append(Hero_agent.get_cumulative_return())
        V_g.append(torch.stack([g.get_cumulative_return() for g in Villans]).mean())

        Hero_agent.reset_saved_stats()
        for g in Villans:
            g.reset_saved_stats()

    sys.stdout.log_unmute()
    V_f_mean = torch.tensor(V_f, device=Hero_agent.device).mean()
    V_g_mean = torch.tensor(V_g, device=Hero_agent.device).mean()
    apply_to_agents(Villans + [Hero_agent], lambda a: a.reset_saved_stats())
    print('ended Nash training')
    print('Vf mean: {:} , Vg mean: {:} '.format(V_f_mean, V_g_mean))

    return torch.tensor(V_f, device=Hero_agent.device).mean()

    # if is_hero_update:
    #
    #     return torch.tensor(V_f, device=Hero_agent.device).mean()
    # else:
    #     return torch.tensor(V_g, device=Hero_agent.device).mean()


def run_game(Hero_agent: MultiAgentSearchAgent, Villains: List[GhostAgent], starting_agent_idx):
    """
    runs the game given the pacman and ghost agents and the starting agent index
    it samples a starting state from the recorded experts games
    :param starting_agent_idx: which agent should start the game
    :param Hero_agent: pacman
    :param Villains: ghosts
    :return: None
    """
    sampled_state = sample_starting_state(starting_agent_idx)
    start_command = ['-p', 'AlphaBetaAgent', '-l', 'smallClassic', '-q']
    args = readCommand(start_command)
    rules = ClassicGameRules()
    game = rules.newGame(sampled_state[0][0].data.layout, Hero_agent, Villains, args['display'],
                         initState=sampled_state[0][0], startingIndex=starting_agent_idx)
    game.run()


def move_to_device(device, agents):
    for agent in agents:
        agent.to(device)


def reward_pretraining(reward_model: RewardApprocimator, shuffles: int):
    with open('./rec_games/all_games.list', 'rb') as f:
        all_games = pickle.load(f)

    reward_model.set_trainable(True)
    for i in range(shuffles):

        shuffle(all_games)
        for g in all_games:
            game_state, state_vec = g[0]
            state_vector = torch.tensor(state_vec.flatten(), device=reward_model.device).float()
            pred = reward_model(state_vector)
            reward_guess = torch.tensor(calc_approximate_reward(game_state), device=reward_model.device).float()

            loss = (reward_guess - pred)**2 #+ F.smooth_l1_loss(pred.squeeze(), torch.tensor([reward_guess],
                                                   #                                               device=reward_model.device))

            reward_model.optimizer.zero_grad()
            loss.backward()

            # if self.clip_grads:
            #     for param in self.parameters():
            #         param.grad.data.clamp_(-1, 1)

            reward_model.optimizer.step()

            print('loss is: ', loss.item())


def IRL():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    R = RewardApprocimator(device=device)
    R_optimizer = torch.optim.Adam(R.parameters(), lr=1e-2)
    R.set_optimizer(R_optimizer)

    sys.stdout.log_mute()
    reward_pretraining(R, 10)
    sys.stdout.log_unmute()

    models_dict = {
        'reward_model': R.state_dict(),
    }
    time_str = time.strftime("%m-%d %H-%M-%S")
    torch.save(models_dict, './checkpoints/rewards_pretratined_' + time_str)



    pacman_agent = OptimalAgent(R=R, device=device)
    pac_optimizer = torch.optim.Adam(pacman_agent.parameters(), lr=1e-2)
    pacman_agent.set_optimizer(pac_optimizer)
    pacman_expert = AlphaBetaAgent(depth=4)

    ghost_experts = [DirectionalGhost(index=1), DirectionalGhost(index=2)]
    ghost_agents = [OptimalGhost(index=1, R=R, device=device), OptimalGhost(index=2, R=R, device=device)]
    for g in ghost_agents:
        opt = torch.optim.Adam(g.parameters(), lr=1e-2)
        g.set_optimizer(opt)

    move_to_device(device, [pacman_agent] + ghost_agents)
    K_G, K_cycle, K_R, I_R, T = 2000, 10000, 2, 200, 1000
    # K_G, K_cycle, K_R, I_R, T = 2, 4, 2, 2, 2
    p = 5
    Nash_TH = 1

    for reward_loop_idx in range(50000):
        pacman_best_value = Nash_GANs_training(pacman_agent, ghost_agents, K_G, K_cycle, R, is_hero_update=True, T=T)
        ghosts_best_value = Nash_GANs_training(pacman_agent, ghost_agents, K_cycle - K_G, K_cycle, R, T=T, is_hero_update=False)

        print("nash sum is :" + str(abs(pacman_best_value - ghosts_best_value)))
        if reward_loop_idx % K_R == 0 and abs(pacman_best_value - ghosts_best_value) < Nash_TH:

            print("REWARD UPDATE step")
            R.set_trainable(True)
            pacman_agent.set_trainable(False)
            pacman_agent.should_record = True
            for g in ghost_agents:
                g.should_record = True
                g.set_trainable(False)

            for update_idx in range(I_R):



                R.set_trainable(True)

                sys.stdout.log_mute()
                run_game(pacman_agent, ghost_experts, 0)
                run_game(pacman_expert, ghost_agents, 0)
                sys.stdout.log_unmute()

                V_f = pacman_agent.get_cumulative_return()
                V_g = torch.stack([g.get_cumulative_return() for g in ghost_agents]).to(pacman_agent.device).mean()

                actual_rewards = torch.tensor(pacman_agent.actual_rewards).to(pacman_agent.device).float().squeeze()
                model_rewards = torch.stack(pacman_agent.rewards).to(pacman_agent.device).squeeze()
                cov_term = cov(model_rewards, actual_rewards)
                cov_coeff = cov_term/(model_rewards.var()*actual_rewards.var())
                mean_term = model_rewards.mean().abs()
                var_term = (model_rewards.var()-p).abs()
                reg_term = cov_term + mean_term + var_term

                loss = V_f + V_g + reg_term  # type: tensor

                print("Vf: {:} , Vg: {:} , cov: {:} , cov_coeff: {:} ,"
                      " mean_term: {:} , var_term: {:}, loss: {:}".format(V_f, V_g,
                                                                          cov_term,
                                                                          cov_coeff,
                                                                          mean_term,
                                                                          var_term,
                                                                          loss))

                R.optimizer.zero_grad()
                loss.backward()
                if R.clip_grads:
                    for param in R.parameters():
                        param.grad.data.clamp_(-1, 1)

                R.optimizer.step()

                pacman_agent.reset_saved_stats()
                for g in ghost_agents:
                    g.reset_saved_stats()

                stats = {'Vf': V_f, 'Vg': V_g, 'cov': cov_term, 'cov_coeff': cov_coeff, 'mean_reward': mean_term}
                if update_idx == I_R - 1:
                    save_checkpoint(pacman_agent, ghost_agents, R, stats)


def cov(a, b):
    if len(a)==0:
        return torch.tensordot(a - a.mean(), b - b.mean(), dims=0)
    return torch.tensordot(a-a.mean(), b-b.mean(), dims=1)


def cov_coeff(a, b):
    return torch.tensordot(a - a.mean(), b - b.mean(), dims=1)/(a.std()*b.std())


def save_checkpoint(Hero: OptimalAgent, Villains: List[OptimalGhost], reward_model: RewardApprocimator, stats: Dict):

    models_dict = {
        'reward_model': reward_model.state_dict(),
        'pac_state_dict': Hero.state_dict()
    }

    for s_k, s_v in stats.items():
        models_dict[s_k] = s_v

    for v in Villains:
        models_dict['villain_{:}_model'.format(v.index)] = v.state_dict()

    time_str = time.strftime("%m-%d %H-%M-%S")
    torch.save(models_dict, './checkpoints/models_Vf {:}_covef {: .3f}_'.format(stats['Vf'], stats['cov_coeff']) + time_str)



def load_checkpoint(path: str):
    pass


if __name__ == '__main__':

    sys.stdout = Logger(logfile)
    IRL()
    #
    # start_command = ['-p', 'AlphaBetaAgent', '-l', 'smallClassic']
    # args = readCommand(start_command)
    # sampled_state = sample_starting_state(0)
    # print(sampled_state[0][0])
    # rules = ClassicGameRules()
    # game = rules.newGame(sampled_state[0][0].data.layout, AlphaBetaAgent(), [RandomGhost(1), RandomGhost(2)], args['display'],
    #                      initState=sampled_state[0][0])
    # game.run()