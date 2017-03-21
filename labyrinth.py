from .moveset import Direction
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import logging
import re

class Labyrinth(object):

    pattern=re.compile('[01]')

    def __init__(self, random_init, rows=5, cols=5, file=None):
        self.rows = int(rows)
        self.cols = int(cols)
        self.movecells = np.zeros((self.rows, self.cols))
        if random_init:
            self._next_cell(0, 0)
            self.array = Labyrinth._cells_to_arr(self.movecells)
        elif file:
            self.array = list()
            with file.open('r') as f:
                for line in f:
                    if re.match(self.pattern, line):
                        self.array.append(line)
                    else:
                        raise Exception('Bad input labytinth file')
            self.array = np.array(self.array)

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
                print('Stopping due to wall encounter, stepping back')
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
            # Calculate next step
            next_path = self._next_cell(next_row, next_col)
        return True

    def get_fitness(self, moveset):
        pass

    def process_moveset(self, moveset):
        distance = 0
        bumps = 0
        redundancy = 0

        return distance, bumps, redundancy


    def plot_arr(self):
        plt.imshow(self.array, cmap=cm.Greys, interpolation='none')
        plt.show()

    @staticmethod
    def _cells_to_arr(movecells, multiplier=10, line_thickness_hor=1, line_thickness_ver=1):

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