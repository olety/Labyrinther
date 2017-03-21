from enum import IntEnum
import random
import numpy as np
import logging


class Direction(IntEnum):
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3
    # top = 00, bot = 01, left = 10, right = 11
    # invert direction : ^1 (xor 1)
    # 00^1 -> 01, etc


class Move():
    def __init__(self, direction):
        self.direction = Direction(direction)

    def mutate(self):
        self.direction = Direction(self.direction ^ 1)

    @property
    def name(self):
        return self.direction.name

    @staticmethod
    def from_string(string):
        return Move(Direction(int(string, 2)))


class Moveset():
    # It's more of a move list :D
    def __init__(self, num_moves=0, string=None):
        if num_moves:
            str_arr = np.array(['{0:02b}'.format(random.getrandbits(2)) for i in range(0, num_moves)])
            self._parse_move_array(str_arr)
        elif string:
            self._parse_move_string(string)
        else:
            raise Exception('Moveset - can\'t create moveset from nothing')

    def mutate(self, num_bits=1, repeating=False):
        # Repeating = True enables mutating one move more than once
        moves_to_mutate = np.arange(self.moveset.size)
        logging.debug('Moveset - Mutating a moveset')
        for i in np.arange(num_bits):
            chosen = random.choice(moves_to_mutate)
            logging.debug('Moveset - Mutating move №{} - {}'.format(chosen, self.moveset[chosen].name))
            if repeating:
                moves_to_mutate.remove(chosen)
            self.moveset[chosen].mutate()
            logging.debug('Moveset - Mutated move №{} into {}'.format(chosen, self.moveset[chosen].name))
        logging.debug('Moveset - New moveset:')
        if logging.DEBUG:
            self.print_moves()

    def enumerated(self):
        return enumerate(self.moveset)

    def _parse_move_string(self, string):
        if string is None or len(string) % 2:
            raise Exception('Moveset - bad string len in _parse_move_string in Moveset')
        iterator_str = iter(string)
        # '00011011' -> ['00', '01', '10', '11']
        logging.debug('Parsing move string {}'.format(string))
        self.moveset = np.array([Move.from_string(character + next(iterator_str, '')) for character in iterator_str])

    def _parse_move_array(self, str_arr):
        if len(str_arr) <= 0:
            raise Exception('Moveset - Bad move array length')
        logging.debug('Moveset - Parsing move array {}'.format(str_arr))
        self.moveset = np.array([Move.from_string(string) for string in str_arr])

    def print_moves(self):
        logging.info('Moveset - Printing moveset ({0} moves)'.format(len(self.moveset)))
        for index, move in enumerate(self.moveset):
            logging.info('\t№{0} - {1}'.format(index, move.direction.name))
        logging.info('Moveset -  Finished printing moveset')

    def __str__(self):
        return ' '.join([move.direction.name for move in self.moveset])

    def __len__(self):
        return len(self.moveset)

    @property
    def num_bits(self):
        return self.moveset.size * 2

    @property
    def move_string(self):
        return ''.join('{0:02b}'.format(move.direction.value) for move in self.moveset)

    @staticmethod
    def crossover(parent1, parent2, num_points):
        if num_points < 0 or num_points > len(parent1) or num_points > len(parent2):
            raise Exception('Moveset - bad number of points for crossover = {}'.format(num_points))
        # crossover_pts = \
        #     [random.randint(range(
        #      len(parent1) if len(parent1) < len(parent2) else len(parent2)))
        #      for i in range(num_points)]
        child = list()
        parents = (parent1, parent2)
        smaller_len = len(parent1) if len(parent1) < len(parent2) else len(parent2)
        crossover_points = sorted([random.randint(0, smaller_len) for i in range(num_points)])
        current_parent = 0
        current_point = 0
        for point in crossover_points:
            child.append(parents[current_parent][current_point:point])
            current_point = point
            current_parent ^= 1


# m = Move(Direction.TOP)
# print('Direction : {0}, binary : {1:02b}, int : {2}'.format(m.direction.name, m.direction.value, m.direction.value))

# mvs = Moveset(string='0011100101')
# mvs.print_moves()

if __name__ == '__main__':
    rand_mvs = Moveset(5)
    rand_mvs.print_moves()
    rand_mvs.mutate()

