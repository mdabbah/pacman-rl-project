from pacman import readCommand, runGames
import os
import sys
from time import time


class timer:

    def __init__(self):
        self.filename = './times_log.txt'
        self.t = 0

    def __enter__(self):
        self.open_file = open(self.filename, 'a')
        self.t = time()
        return self.open_file

    def __exit__(self, *args):

        # self.open_file.write('\n' + str(self.t-time()))
        self.open_file.close()


logfile = 'log.txt'
std_out_orig = sys.stdout


class Logger(object):
    def __init__(self, file_name):
        self.terminal = std_out_orig
        self.log = file_name

    def write(self, message):
        self.terminal.write(message)
        with open(self.log, 'a') as f:
            f.write(message)


if __name__ == '__main__':

    os.makedirs('stats', exist_ok=True)
    players = ['DirectionalExpectimaxAgent']

    # ['ReflexAgent']  #  ['MinimaxAgent', 'AlphaBetaAgent', 'RandomExpectimaxAgent']

    # layouts = os.listdir(r'./layouts')
    # layouts = [os.path.splitext(l)[0] for l in layouts]  # ['mediumClassic']#

    layouts = ['trickyClassic']
    ghosts = ['DirectionalGhost', 'RandomGhost']

    depth = [4]  #[2, 3, 4]

    eval_functions = ['betterEvaluationFunction']
    num_games = 5

    for p in players:

        for l in layouts:

            for g in ghosts:

                if p == 'ReflexAgent':
                    depth = [1]


                for d in depth:

                    command = ['-p', p, '-l', l, '-n', str(num_games), '-k', '2', '-g', g,
                               '-a', 'depth={:}'.format(d)]

                    agent = p
                    # if p == 'ReflexAgent':
                    #     command = ['-p', p, '-l', l, '-n', str(num_games), '-k', '2']
                    #     # agent = 'BetterAgent'
                    #
                    if '-q' in sys.argv:
                        command.append('-q')

                    file_name_times = os.path.join('stats', 'times_{:}_{:}_{:}_{:}.txt'.format(agent, l, d, g))
                    file_name = os.path.join('stats', '{:}_{:}_{:}_{:}.txt'.format(agent, l, d, g))
                    sys.stdout = Logger(file_name)
                    print('running command: ' + str(command))
                    args = readCommand(command)  # Get game components based on input
                    runGames(**args)
                    os.rename('./stats/times_log.txt', file_name_times)
