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


class Move(object):

    def __init__(self, direction):
        self.direction = Direction(direction)

    def mutate(self):
        self.direction = Direction(self.direction^1)

    @property
    def name(self):
        return self.direction.name

    @staticmethod
    def from_string(string):
        return Move(Direction(int(string, 2)))


class Moveset(object):
    # It's more of a move list :D
    def __init__(self, num_moves=0, string=None):
        if num_moves:
            str_arr = np.array(['{0:02b}'.format(random.getrandbits(2)) for i in range(0, num_moves)])
            self._parse_move_array(str_arr)
        else:
            self._parse_move_string(string)

    def mutate(self, num_bits=1, repeating=False):
        # Repeating = True enables mutating one move more than once
        moves_to_mutate = np.arange(self.moveset.size)
        logging.debug('Moveset - Mutating a moveset')
        for i in np.arange(num_bits):
            chosen = random.choice(moves_to_mutate)
            logging.debug('Moveset - Mutating move #{} - {}'.format(chosen, self.moveset[chosen].name))
            if repeating:
                moves_to_mutate.remove(chosen)
            self.moveset[chosen].mutate()
        logging.debug('Moveset - New moveset:')
        if logging.DEBUG:
            self.print_moves()

    def _parse_move_string(self, string):
        if string is None or len(string) % 2:
            raise Exception('bad string len in _parse_move_string in Moveset')
        iterator_str = iter(string)
        # '00011011' -> ['00', '01', '10', '11']
        logging.debug('Parsing move string {}'.format(string))
        self.moveset = np.array([Move.from_string(character + next(iterator_str, '')) for character in iterator_str])

    def _parse_move_array(self, str_arr):
        if len(str_arr) <= 0:
            raise Exception('bad move array length')
        logging.debug('Parsing move array {}'.format(str_arr))
        self.moveset = np.array([Move.from_string(string) for string in str_arr])

    def print_moves(self):
        logging.info('-- Printing moveset ({0} moves) --'.format(len(self.moveset)))
        for index, move in enumerate(self.moveset):
            logging.info('\t№{0} - {1}'.format(index+1, move.direction.name))
        logging.info('-- Finished printing moveset --')

    @property
    def num_bits(self):
        return self.moveset.size * 2

    @property
    def move_string(self):
        return ''.join('{0:02b}'.format(move.direction.value) for move in self.moveset)

# m = Move(Direction.TOP)
# print('Direction : {0}, binary : {1:02b}, int : {2}'.format(m.direction.name, m.direction.value, m.direction.value))

# mvs = Moveset(string='0011100101')
# mvs.print_moves()

logging.basicConfig(level=logging.DEBUG)
rand_mvs = Moveset(5)
rand_mvs.print_moves()
rand_mvs.mutate()

