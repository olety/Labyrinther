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


class Move:
    def __init__(self, direction):
        self.direction = Direction(direction)

    def mutate(self):
        self.direction = Direction(self.direction ^ random.randint(1, 2))

    def __str__(self):
        return self.direction.name

    @property
    def name(self):
        return self.direction.name

    @staticmethod
    def from_string(string):
        return Move(Direction(int(string, 2)))


class Moveset:
    # It's more of a move list :D
    def __init__(self, num_moves=0, string=None, move_list=None):
        if num_moves:
            str_arr = np.array(['{0:02b}'.format(random.getrandbits(2)) for i in range(0, num_moves)])
            self._parse_move_array(str_arr)
        elif string:
            self._parse_move_string(string)
        elif move_list.all():
            self.move_list = move_list
        else:
            raise Exception('Moveset - can\'t create moveset from nothing')

    def mutate(self, num_bits=1, repeating=False):
        # Repeating = True enables mutating one move more than once
        moves_to_mutate = list(range(len(self.move_list)))
        logging.debug('Moveset - Mutating a moveset')
        # Doing it this way  because I think that it's faster for longer moves than changing string and reiniting
        for i in range(num_bits):
            chosen = random.choice(moves_to_mutate)
            # print(chosen, self.move_list[chosen], self.move_list)
            # logging.debug('Moveset - Mutating move №{} - {}'.format(chosen, self.move_list[chosen].name))
            if repeating:
                moves_to_mutate.remove(chosen)
            self.move_list[chosen].mutate()
            # logging.debug('Moveset - Mutated move №{} into {}'.format(chosen, self.move_list[chosen].name))
        # logging.debug('Moveset - New moveset:')
        # if logging.DEBUG:
        #     self.print_moves()

    def enumerated(self):
        return enumerate(self.move_list)

    def _parse_move_string(self, string):
        if string is None or len(string) % 2:
            raise Exception('Moveset - bad string len in _parse_move_string in Moveset')
        iterator_str = iter(string)
        # '00011011' -> ['00', '01', '10', '11']
        logging.debug('Parsing move string {}'.format(string))
        self.move_list = np.array([Move.from_string(character + next(iterator_str, '')) for character in iterator_str])

    def _parse_move_array(self, str_arr):
        if len(str_arr) <= 0:
            raise Exception('Moveset - Bad move array length')
        logging.debug('Moveset - Parsing move array {}'.format(str_arr))
        self.move_list = np.array([Move.from_string(string) for string in str_arr])

    def print_moves(self):
        print('Moveset - Printing moveset ({0} moves)'.format(len(self.move_list)))
        for index, move in enumerate(self.move_list):
            print('\t№{0} - {1}'.format(index, move.direction.name))
        print('Moveset -  Finished printing moveset')

    def __str__(self):
        return ' '.join([move.direction.name for move in self.move_list])

    def __len__(self):
        return len(self.move_list)

    @property
    def num_bits(self):
        return self.move_list.size * 2

    @property
    def move_string(self):
        return ''.join('{0:02b}'.format(move.direction.value) for move in self.move_list)

    def crossover(self, parent2, num_points):
        logging.debug('Moveset - Crossover - self len = {}, p2 len = {}, num pts = {}'.format(len(self), len(parent2), num_points))
        if num_points < 0 or num_points > len(self):
            raise Exception('Moveset - bad number of points for crossover = {}'.format(num_points))
        child = np.empty(0, dtype=object)
        parents = (self.move_list, parent2.move_list)
        if len(self) < len(parent2):
            rand_len = len(self)
            fill_pts = False
        else:
            rand_len = len(parent2)
            fill_pts = True
        crossover_points = sorted([random.randint(0, rand_len) for i in range(num_points)])
        current_parent = 0
        current_point = 0
        for point in crossover_points:
            child = np.concatenate([child, parents[current_parent][current_point:point+1]])
            current_point = point+1
            current_parent ^= 1
        child = np.concatenate([child, parents[current_parent][current_point:]])
        if fill_pts:
            child = np.concatenate([child, self.move_list[len(child):]])
        return Moveset(move_list=child)


# m = Move(Direction.TOP)
# print('Direction : {0}, binary : {1:02b}, int : {2}'.format(m.direction.name, m.direction.value, m.direction.value))

# mvs = Moveset(string='0011100101')
# mvs.print_moves()

if __name__ == '__main__':
    # Just testing different things
    rand_mvs = Moveset(5)
    print([item.name for item in rand_mvs.move_list])
    print(rand_mvs.move_string)
    # rand_mvs.mutate()

