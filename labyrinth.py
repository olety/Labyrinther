from moveset import Direction
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import logging
import re


class Labyrinth(object):

    def __init__(self, random_init=False, rows=5, cols=5, file=None):
        if random_init is False and file is None:
            raise Exception('Can\'t make a labyrinth out of nothing')
        logging.debug('Labyrinth - started processing labyrinth')
        logging.debug('Labyrinth - Shape: [{},{}], random = {}, file={}'
                      .format(rows, cols, random_init, file))
        if random_init is True:
            self.rows = int(rows)
            self.cols = int(cols)
            self.movecells = np.zeros((self.rows, self.cols, 5))
            self._next_cell(0, 0)
            logging.debug('Labyrinth - Generated a labyrinth')
            self.array_closed = Labyrinth._cells_to_arr(self.movecells)
            self.array_open = Labyrinth._cells_to_arr(self.movecells, True)
            logging.debug('Labyrinth - converted movecells to array')
        elif file:
            self._from_file(file)
        # Default multiplier is 10 so we have to divide by 10**2
        logging.debug('Labyrinth - array shape is {}'.format(self.array_closed.shape))
        self.num_moves_max = (self.array_closed.shape[0] * self.array_closed.shape[1]) / 100
        logging.debug('Labyrinth - num_moves_max = {}'.format(self.num_moves_max))

    def _from_file(self, file):
        with open(file, 'r') as f:
            self.rows, self.cols = [int(a) for a in f.readline().split(' ')]
            logging.debug('Labyrinth - file - Found shape: [{},{}]'.format(self.rows, self.cols))
        temp_df = pd.read_csv(file, sep=' ', header=None, skiprows=1)
        self.array_closed = temp_df.values
        if len(np.unique(self.array_closed)) != 2:
            logging.error('Labyrinth - file didn\'t match the pattern')
            raise Exception('Bad input labyrinth file')
        self.array_open = self.array_closed.copy()
        self.array_open[1:8, 0] = 0
        self.array_open[self.rows * 10 - 9:self.rows * 10 - 2, self.cols * 10 - 1] = 0

    def _next_cell(self, row, col):
        # assign values to cells and set cells[row,col,4] - visited to 1
        logging.debug('Labyrinth - init - at node [{},{}]'.format(row, col))
        self.movecells[row, col, 4] = 1
        next_path = False
        # Entering a loop
        while next_path is False:
            choices = list()
            # Filling choices array
            if row > 1 and self.movecells[row - 1, col, 4] == 0:
                choices.append(Direction.TOP)
            if row < self.rows - 1 and self.movecells[row + 1, col, 4] == 0:
                choices.append(Direction.BOTTOM)
            if col > 1 and self.movecells[row, col - 1, 4] == 0:
                choices.append(Direction.LEFT)
            if col < self.cols - 1 and self.movecells[row, col + 1, 4] == 0:
                choices.append(Direction.RIGHT)
            # Exit condition
            if len(choices) < 1:
                logging.debug('Labyrinth - generator - Stopping due to wall encounter, stepping back')
                return False
            # Choose a direction
            chosen_dir = random.choice(choices)
            choices.remove(chosen_dir)
            self.movecells[row, col, chosen_dir] = 1
            next_row = row
            next_col = col
            if chosen_dir == Direction.TOP:
                next_row -= 1
            elif chosen_dir == Direction.BOTTOM:
                next_row += 1
            elif chosen_dir == Direction.LEFT:
                next_col -= 1
            elif chosen_dir == Direction.RIGHT:
                next_col += 1
            logging.debug('Labyrinth - init - at node [{},{}], going {} -> next [{},{}]'
                          .format(row, col, chosen_dir.name,next_row, next_col))
            # If we go somewhere we can go back
            self.movecells[next_row, next_col, chosen_dir ^ 1] = 1
            # Go there and calculate next step
            next_path = self._next_cell(next_row, next_col)
        return True

    def get_fitness(self, moveset):
        pass

    def save_file(self, name):
        # with open(name, 'w') as f:
        #     f.writelines(
        with open(name+'.csv', 'w') as f:
            f.write('{} {}\n'.format(self.rows, self.cols))
        with open(name+'.csv', 'a') as f:
            pd.DataFrame(data=self.array_closed.astype(int))\
                .to_csv(f, sep=' ', header=False, index=False)

    def process_moveset(self, moveset):
        distance = 0
        bumps = 0
        redundancy = 0
        for index, move in moveset.enumerated():
            #move.direction
            pass
        return distance, bumps, redundancy

    def print_arr(self, open_ends=True):
        print('Printing labytinth array, open={}'.format(open_ends))
        if open_ends:
            print(self.array_open)
        else:
            print(self.array_closed)

    def plot_arr(self, open_ends=True):
        print('Plotting labytinth array, open={}'.format(open_ends))
        if open_ends:
            plt.imshow(self.array_open, cmap=cm.Greys, interpolation='none')
        else:
            plt.imshow(self.array_closed, cmap=cm.Greys, interpolation='none')
        plt.show(block=False)

    @staticmethod
    def _cells_to_arr(movecells, open_endpoints=False, multiplier=10, line_thickness_hor=1, line_thickness_ver=1):
        movecells = np.copy(movecells)
        if open_endpoints:
            movecells[0, 0, 2] = 1
            movecells[movecells.shape[1]-1, movecells.shape[0]-1, 3] = 1
        width = multiplier * movecells.shape[1]
        cell_width = multiplier - line_thickness_hor * 2
        height = multiplier * movecells.shape[0]
        cell_height = multiplier - line_thickness_ver * 2
        dup = np.ones((height, width))
        for ver in range(movecells.shape[0]):
            for hor in range(movecells.shape[1]):
                ver_start = ver * multiplier + line_thickness_ver
                hor_start = hor * multiplier + line_thickness_hor
                if movecells[ver, hor, Direction.LEFT] == 1:
                    dup[ver_start:ver_start + cell_height,
                        hor_start - line_thickness_hor:hor_start + cell_width] = 0
                if movecells[ver, hor, Direction.RIGHT] == 1:
                    dup[ver_start:ver_start + cell_height,
                        hor_start:hor_start + cell_width + line_thickness_hor] = 0
                if movecells[ver, hor, Direction.TOP] == 1:
                    dup[ver_start - line_thickness_ver:ver_start + cell_height,
                        hor_start:hor_start + cell_width] = 0
                if movecells[ver, hor, Direction.BOTTOM] == 1:
                    dup[ver_start:ver_start + cell_height + line_thickness_ver,
                        hor_start:hor_start + cell_width] = 0
        return dup

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    np.set_printoptions(threshold=np.nan)
    lab = Labyrinth(file='lab_test.csv')
    # lab = Labyrinth(random_init=True)
    # lab.plot_arr()
    # lab.print_arr()
    # lab.print_arr(False)
    # lab.save_file('lab_test')
    # plt.show()

