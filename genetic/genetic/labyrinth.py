from .moveset import Direction, Moveset
import random
import math
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib import cm
import logging


class Labyrinth:
    def __init__(self, rows=5, cols=5, file=None, file_obj=None):
        if (not rows or not cols) and file is None:
            raise Exception('Can\'t make a labyrinth out of nothing')
        logging.debug('Labyrinth - started processing labyrinth')
        logging.debug('Labyrinth - Shape: [{},{}], file={}'
                      .format(rows, cols, file))
        if file_obj:
            self._from_file_obj(file_obj)
        if file:
            self._from_file(file)
        else:
            self.rows = int(rows)
            self.cols = int(cols)
            self.movecells = np.zeros((self.rows, self.cols, 5))
            self._next_cell(0, 0)
            logging.debug('Labyrinth - Generated a labyrinth')
            self.array_closed = Labyrinth._cells_to_arr(self.movecells)
            self.array_open = Labyrinth._cells_to_arr(self.movecells, True)
            logging.debug('Labyrinth - converted movecells to array')

        logging.debug('Labyrinth - array shape is {}'.format(self.array_closed.shape))
        # Default multiplier is 10 so we have to divide by 10**2
        self.num_moves_max = self.array_closed.shape[0] * self.array_closed.shape[1] / 100
        logging.debug('Labyrinth - num_moves_max = {}'.format(self.num_moves_max))

    def _from_file_obj(self, file):
        self.rows, self.cols = (int(x) for x in file.split(' ')[0:1])
        temp_df = pd.read_csv(file, sep='-', header=None, skiprows=1)
        self.array_closed = temp_df.values
        # print(self.array_closed)
        if len(np.unique(self.array_closed)) != 2:
            logging.error('Labyrinth - file didn\'t match the pattern')
            raise Exception('Bad input labyrinth file')
        self.array_open = self.array_closed.copy()
        self.array_open[1:8, 0] = 0
        self.array_open[self.rows * 10 - 9:self.rows * 10 - 2, self.cols * 10 - 1] = 0
        self.movecells = Labyrinth._arr_to_cells(self.array_open, self.rows, self.cols)

    def _from_file(self, file):
        with open(file, 'r') as f:
            try:
                self.rows, self.cols = [int(a) for a in f.readline().split(' ')]
            except Exception:
                raise Exception('Bad input labyrinth file')
            logging.debug('Labyrinth - file - Found shape: [{},{}]'.format(self.rows, self.cols))
        temp_df = pd.read_csv(file, sep='-', header=None, skiprows=1)
        self.array_closed = temp_df.values
        # print(self.array_closed)
        if len(np.unique(self.array_closed)) != 2:
            logging.error('Labyrinth - file didn\'t match the pattern')
            raise Exception('Bad input labyrinth file')
        self.array_open = self.array_closed.copy()
        self.array_open[1:8, 0] = 0
        self.array_open[self.rows * 10 - 9:self.rows * 10 - 2, self.cols * 10 - 1] = 0
        self.movecells = Labyrinth._arr_to_cells(self.array_open, self.rows, self.cols)

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
                          .format(row, col, chosen_dir.name, next_row, next_col))
            # If we go somewhere we can go back
            self.movecells[next_row, next_col, chosen_dir ^ 1] = 1
            # Go there and calculate next step
            next_path = self._next_cell(next_row, next_col)
        return True

    def save_file(self, name):
        # with open(name, 'w') as f:
        #     f.writelines(
        with open(name + '.csv', 'w') as f:
            f.write('{} {}\n'.format(self.rows, self.cols))
        with open(name + '.csv', 'a') as f:
            pd.DataFrame(data=self.array_closed.astype(int)) \
                .to_csv(f, sep='-', header=False, index=False)

    def plot_moveset(self, moveset, show=True, savefig=False, file_name='', **kwargs):
        movecells = self.movecells.copy()
        path = [[4.5, 4.5]]
        bumps = []
        row = 0
        col = 0
        logging.info('Labyrinth - Processing moveset: {}'.format(moveset))
        for index, move in moveset.enumerated():
            if row >= self.rows or col >= self.cols:
                continue
            if movecells[row, col, move.direction.value] == 1:
                # print('Going {}'.format(move.direction.name))
                if move.direction == Direction.TOP:
                    row -= 1
                elif move.direction == Direction.BOTTOM:
                    row += 1
                elif move.direction == Direction.LEFT:
                    col -= 1
                elif move.direction == Direction.RIGHT:
                    col += 1
                path.append([row * 10 + 4.5, col * 10 + 4.5])
            else:
                bumps.append([row * 10 + 4.5, col * 10 + 4.5])

        if kwargs.get('export_data', False):
            return path, bumps
        return self._plot_moveset(path=path, bumps=bumps, show=show, savefig=savefig, file_name=file_name, **kwargs)

    @staticmethod
    def _transform_array_into_xy(arr):
        # Array format: [(y1, x1), ...]
        return [p[1] for p in arr], [p[0] for p in arr]

    @staticmethod
    def _transform_array_into_xy_stack(arr):
        # Array format: [(y1, x1), ...]
        return np.hstack([p[1] for p in arr], [p[0] for p in arr])

    def _plot_moveset(self, path, bumps, **kwargs):
        if kwargs.get('omit_figures', False) and kwargs.get('ax', None):
            ax = kwargs['ax']
        else:
            f, ax = plt.subplots()
            ax.imshow(self.array_closed, cmap=cm.Greys, origin='upper', zorder=0)

        path_xs, path_ys = Labyrinth._transform_array_into_xy(path)
        bumps_xs, bumps_ys = Labyrinth._transform_array_into_xy(bumps)
        # Plotting
        if kwargs.get('plot', False):
            plots = list()
            plots.append(ax.scatter(bumps_xs, bumps_ys, color='#ff0000', s=(1800 - self.rows * self.cols * 12), alpha=0.1,
                                    marker='^',
                                    zorder=1))
            plots.append(
                ax.scatter(path_xs[0], path_ys[0], color='#ff6a00', s=(2000 - self.rows * self.cols * 15), marker='o',
                           alpha=1, zorder=2, label='start'))
            plots.append(
                ax.scatter(path_xs[-1], path_ys[-1], color='#c447e0', s=(2000 - self.rows * self.cols * 15), marker='X',
                           alpha=1, zorder=2, label='finish'))
            # Marking the goal
            ax.scatter(self.rows * 8 + 4.5, self.cols * 8 + 4.5, color='y', s=(1000 - self.rows * self.cols * 15),
                           marker='D',
                           alpha=1, zorder=2, label='goal')

            for i in range(len(path) - 2, -1, -1):
                p = FancyArrowPatch(posA=[path_xs[i], path_ys[i]], posB=[path_xs[i + 1], path_ys[i + 1]],
                                    connectionstyle='arc3, rad=0.5',
                                    arrowstyle='simple, head_width={}, head_length={}'
                                    .format(30 - (self.rows + self.cols), 25 - (self.rows + self.cols)),
                                    edgecolor='#5f5d63',
                                    facecolor='#42c8f4',
                                    zorder=2 + i)
                ax.add_artist(p)
                plots.append(p)
            plots.append(ax.scatter(path_xs[i + 1], path_ys[i + 1], zorder=2 + i, color='#42c8f4'))
            plt.axis('off')
            plt.title('Moveset')
            plt.title('{}x{}'.format(self.rows, self.cols), loc='right')
            plt.title(kwargs.get('left_title', ''), loc='left')
        return_dict = dict()
        if kwargs.get('show', False):
            plt.show()
        if kwargs.get('savefig', False) and kwargs.get('file_name', ''):
            f.savefig(kwargs['file_name'], dpi=500)
        if kwargs.get('export', False):
            return_dict['fig'] = f
            return_dict['ax'] = ax
        if kwargs.get('export_plots', False):
            return_dict['plots'] = plots
        return return_dict

    def plot_anim(self, movesets, **kwargs):
        # DOESNT WORK SINE THE LENGTH OF A MOVESET IS VARIABLE
        # setup = self.plot_moveset(movesets[0], show=False, export=True, export_plots=True, plot=True)
        # fig = setup['fig']
        # ax = setup['ax']
        # plots = setup['plots']
        #
        # def update(i):
        #     label = 'Generation {}'.format(i)
        #     path, bumps = self.plot_moveset(movesets[i], export_data=True)
        #     path_xs, path_ys = Labyrinth._transform_array_into_xy(path)
        #     bumps_xs, bumps_ys = Labyrinth._transform_array_into_xy(bumps)
        #     # plots[0].set_offsets(bumps_xs, bumps_ys)
        #     # plots[1].set_offsets(path[0])
        #     # plots[2].set_offsets((path_xs[-1], path_ys[-1]))
        #     for i, j in enumerate(range(len(path)-2, -1, -1), 3):
        #         plots[i].set_positions([path_xs[i], path_ys[i]], [path_xs[i + 1], path_ys[i + 1]])
        #     # plots[len(plots)-1]
        #     return plots
        #
        # mywriter = animation.FFMpegWriter()
        # ani = animation.FuncAnimation(fig, update, frames=np.arange(0, len(movesets)), interval=200)
        # ani.save('asd.mp4', writer=mywriter)

    def process_moveset(self, moveset):
        movecells = self.movecells.copy()
        movecells[:, :, 4] = 0  # Clearing visit history
        bumps = 0
        redundancy = 0
        row = 0
        col = 0

        logging.info('Labyrinth - Processing moveset: {}'.format(moveset))
        for index, move in moveset.enumerated():
            if row >= self.rows or col >= self.cols:
                bumps += 1
                continue

            if movecells[row, col, 4] == 1:
                redundancy += 1

            movecells[row, col, 4] = 1

            if movecells[row, col, move.direction.value] == 0:
                bumps += 1
            else:
                if move.direction == Direction.TOP:
                    row -= 1
                elif move.direction == Direction.BOTTOM:
                    row += 1
                elif move.direction == Direction.LEFT:
                    col -= 1
                elif move.direction == Direction.RIGHT:
                    col += 1

        moves = len(moveset) - bumps
        distance = self.rows - row + self.cols - col - 2
        return distance, bumps, redundancy, moves

    def print_arr(self, open_ends=True):
        logging.info('Printing labyrinth array, open={}'.format(open_ends))
        if open_ends:
            print(self.array_open)
        else:
            print(self.array_closed)

    def print_movecells(self):
        logging.info('Printing labyrinth movecells')
        for row_num, cols in enumerate(self.movecells):
            for col_num, cell in enumerate(cols):
                print('Cell [{},{}]: '.format(row_num, col_num), end=' ')
                for direction in Direction:
                    if cell[direction]:
                        print('{}'.format(direction.name), end=' ')
                print()

    def plot_arr(self, arr, show=True, block=True):
        plt.imshow(arr, cmap=cm.Greys, interpolation='none')
        if show:
            plt.show(block=block)

    @staticmethod
    def _arr_to_cells(array, rows, cols, open_exit=False):
        logging.debug('Labyrinth - Converting array [{},{}] to cells. Arr shape: [{}]'.format(rows, cols, array.shape))
        multiplier = array.shape[1] // rows
        logging.debug('Labyrinth - array conversion multiplier is {}'.format(multiplier))
        movecells = np.zeros((rows, cols, 5))
        for row in range(rows):
            for col in range(cols):
                movecells[row, col, 4] = 1  # We visited it
                if row > 0 and \
                                array[row * multiplier,
                                      col * multiplier - 1 + multiplier // 2] == 0:
                    movecells[row, col, Direction.TOP] = 1

                if row < rows - 1 and \
                                array[(row + 1) * multiplier,
                                      col * multiplier - 1 + multiplier // 2] == 0:
                    movecells[row, col, Direction.BOTTOM] = 1

                if col > 0 and \
                                array[row * multiplier - 1 + multiplier // 2,
                                      col * multiplier] == 0:
                    movecells[row, col, Direction.LEFT] = 1

                if col < cols - 1 and \
                                array[row * multiplier - 1 + multiplier // 2,
                                      (col + 1) * multiplier] == 0:
                    movecells[row, col, Direction.RIGHT] = 1
        if open_exit:
            logging.debug('Labyrinth - opening movecell exit')
            movecells[rows - 1, cols - 1, Direction.RIGHT] = 1
        # movecells = np.flipud(movecells)
        return movecells

    def __str__(self):
        return '{}x{} Labyrinth. {}x multiplier is used in a file representation. num_moves_max = {}' \
            .format(self.rows, self.cols, 10, self.num_moves_max)

    @staticmethod
    def _cells_to_arr(movecells, open_endpoints=False, multiplier=10, line_thickness_hor=1, line_thickness_ver=1):
        movecells = np.copy(movecells)
        if open_endpoints:
            movecells[0, 0, 2] = 1
            movecells[movecells.shape[1] - 1, movecells.shape[0] - 1, 3] = 1
        width = multiplier * movecells.shape[1]
        cell_width = multiplier - line_thickness_hor * 2
        height = multiplier * movecells.shape[0]
        cell_height = multiplier - line_thickness_ver * 2
        dup = np.ones((height, width))
        for ver in range(movecells.shape[0]):
            for hor in range(movecells.shape[1]):
                ver_start = ver * multiplier + line_thickness_ver
                hor_start = hor * multiplier + line_thickness_hor
                if movecells[ver, hor, Direction.TOP] == 1:
                    dup[ver_start - line_thickness_ver:ver_start + cell_height,
                    hor_start:hor_start + cell_width] = 0
                if movecells[ver, hor, Direction.BOTTOM] == 1:
                    dup[ver_start:ver_start + cell_height + line_thickness_ver,
                    hor_start:hor_start + cell_width] = 0
                if movecells[ver, hor, Direction.LEFT] == 1:
                    dup[ver_start:ver_start + cell_height,
                    hor_start - line_thickness_hor:hor_start + cell_width] = 0
                if movecells[ver, hor, Direction.RIGHT] == 1:
                    dup[ver_start:ver_start + cell_height,
                    hor_start:hor_start + cell_width + line_thickness_hor] = 0
        return dup


if __name__ == '__main__':
    # Just testing stuff
    logging.basicConfig(level=logging.DEBUG)
    np.set_printoptions(threshold=np.nan)
    lab = Labyrinth(file='3x3.csv')
    lab.plot_moveset(Moveset(10))
    # if not np.equal(Labyrinth._cells_to_arr(lab.movecells, False), lab.array_closed).all():
    #     print('Bad')
    # else:
    #     print('Good')
    # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    # ax1.imshow(lab.array_closed)
    # ax2.imshow(Labyrinth._cells_to_arr(lab.movecells, False))
    # plt.show()
    # lab = Labyrinth(file='.csv')
    # mvs = Moveset(num_moves=random.randint(lab.num_moves_max//2, lab.num_moves_max))
    # print(lab.process_moveset(mvs))
    # lab.print_movecells()
    # lab = Labyrinth(random_init=True, rows=2, cols=2)
    # lab.plot_arr()
    # lab.print_arr()
    # lab.print_arr(False)
    # lab.save_file('2x2')
    # plt.show()
