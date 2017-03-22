from labyrinth import Labyrinth
from moveset import Moveset
import numpy as np
import pandas as pd
from math import ceil
import random
import logging


class GeneticAlgorithm:
    def __init__(self, labyrinth, num_population=100, min_moves_mult=0.5, max_moves_mult=3, max_iter=100, elitism_num=5,
                 crossover_rate=0.8, crossover_pts='random', mutation_rate=0.001, selection='roulette', **kwargs):
        # Setting the variables
        self.labyrinth = labyrinth
        # Numpy
        # self.all_pops = np.empty((max_iter, num_population, 2))  # generation -> person -> moveset/fitness
        # self.pop = np.empty((num_population, 2), dtype=[('moveset', Moveset), ('fitness', float)])
        # self.next_pop = np.empty((num_population, 2), dtype=[('moveset', Moveset), ('fitness', float)])

        # General
        if 'file_name' in kwargs:
            self.file_name = kwargs['file_name']
        else:
            self.file_name = 'last_run'
        # Algorithm - related
        self.min_moves_mult = min_moves_mult
        self.max_moves_mult = max_moves_mult
        self.max_iter = max_iter
        self.pop_cnt = num_population
        self.elites_cnt = elitism_num
        if 'roulette_mult' in kwargs:
            self.roulette_mult = kwargs['roulette_mult']
        else:
            self.roulette_mult = 1
        if crossover_pts == 'random':
            crossover_pts = random.randint(1, 3)
        self.crossover_rate = crossover_rate
        self.crossover_pts = crossover_pts
        self.mutation_rate = mutation_rate
        # Generating the starting population
        self._generate_initial_pop()
        # Doing selection
        if selection == 'roulette':
            self.winner_gen = self._roulette_selection()
        elif selection == 'roulette-normalized':
            self._normalized_roulette_selection()
        elif selection == 'tournament':
            self._tournament_selection()
        elif selection == 'steady-state':
            self._steady_state_selection()
        else:
            raise Exception('GA - Wrong selection method: {}'.format(selection))

    def _generate_initial_pop(self):
        self.pop = np.array([[Moveset(random.randint(
                            int(self.labyrinth.num_moves_max * self.min_moves_mult),
                            int(self.labyrinth.num_moves_max * self.max_moves_mult))), 0]
                            for _ in range(self.pop_cnt)])
        self.next_pop = self.pop.copy()
        self.all_pops = self.pop.copy()

    @staticmethod
    def _fitness(distance, bumps, redundancy, moves, length):
        return 0.3*moves + 0.5*length - redundancy - 10 * distance - bumps

    def _roulette_selection(self):
        logging.info('Starting roulette selection')
        roulette = np.empty(self.pop_cnt*self.roulette_mult, dtype=object)
        logging.debug('Inited empty roulette with size {}'.format(roulette.shape))
        self.found_winner = False
        for gen in range(self.max_iter):
            logging.info('Generation №{}'.format(gen))
            # Rank the population
            logging.debug('Processing generation №{}'.format(gen))
            logging.debug('Starting initial population fitness eval')
            for i, person in enumerate(self.pop):
                distance, bumps, redundancy, moves = self.labyrinth.process_moveset(person[0])
                person[1] = GeneticAlgorithm._fitness(distance, bumps, redundancy, moves, len(person[0]))
                if distance == 0:
                    logging.info('GA - gen {} - Acquired a winner - moveset {}'.format(gen, person[0]))
                    print('GA - gen {} - Acquired a winner - moveset {}'.format(gen, person[0]))
                    self.found_winner = True
                    self.winner_moveset = person[0]
                    self.pop[:, 1] += -1*np.min(self.pop[:,1])
                    self.winner_fitness = person[1] + -1*np.min(self.pop[:,1])
                    self.all_pops = np.vstack((self.all_pops, self.pop))
                    return gen
            # Sorting the pop and adjusting the fitness to be always >0
            logging.info('Sorting the population')
            self.pop = self.pop[np.flipud(self.pop[:, 1].argsort())]
            self.pop[:, 1] += -1*np.min(self.pop[:,1])
            sum_fitness = np.sum(self.pop[:, 1])
            logging.info('Gen {} fitness sum = {}'.format(gen, sum_fitness))
            logging.info('Gen {} fitness avg = {}'.format(gen, np.mean(self.pop[:, 1])))
            # Determine best ones - Make the roulette
            logging.info('Filling in the roulette')
            curr_pt = 0
            for person in self.pop:
                if person[1] < 0:
                    person[1] = 0
                part = ceil((person[1] / sum_fitness) * len(roulette))
                if curr_pt + part >= len(roulette):
                    part = len(roulette) - curr_pt
                roulette[curr_pt:curr_pt + part] = np.repeat(person[0], part)
                curr_pt += part
            logging.debug('Filled roulette: {}'.format(roulette))
            # Elitism
            logging.info('Filling in the elites')
            self.next_pop[0:self.elites_cnt][0] = self.pop[0:self.elites_cnt][0]
            # Perform crossover
            logging.info('Processing crossover')
            for i, person in enumerate(self.pop[:-self.elites_cnt]):
                if random.uniform(0, 1) <= self.crossover_rate:
                    self.next_pop[self.elites_cnt + i][0] = \
                        person[0].crossover(random.choice(roulette), self.crossover_pts)
                else:
                    self.next_pop[self.elites_cnt + i][0] = person[0]
            # Perform mutations
            logging.info('Processing mutations')
            for person in self.next_pop:
                if random.uniform(0, 1) <= self.mutation_rate:
                    person[0].mutate()
            # Save the values for the history view
            logging.info('Saving the values and moving onto the next gen')
            self.all_pops = np.vstack((self.all_pops, self.pop))
            self.pop = self.next_pop
        return -1

    def print_result(self):
        if self.found_winner:
            print('Found a winner moveset! It was in gen {}, fitness: {}, moveset:{}'
                  .format(self.winner_gen, self.winner_fitness, self.winner_moveset))
        else:
            print('This algorithm didn\'t find a winner yet :(')

    def save_data(self):
        if self.found_winner:
            self.all_pops = self.all_pops[self.pop_cnt:].reshape((self.winner_gen+1, self.pop_cnt, 2))
        else:
            self.all_pops = self.all_pops[self.pop_cnt:].reshape((self.max_iter, self.pop_cnt, 2))
        np.save(self.file_name, self.all_pops)

    @staticmethod
    def analyze(filename):
        all_pops = np.load(filename+'.npy')
        max = -100000000
        max_person = None
        for i, gen in enumerate(all_pops):
            for j, person in enumerate(gen):
                if person[1] >= max:
                    max = person[1]
                    max_person = person[0]
        # print(all_pops)
        print(max, max_person)


    def _normalized_roulette_selection(self):
        pass

    def _tournament_selection(self):
        pass

    def _steady_state_selection(self):
        pass

def run(num_tests):
    lab = Labyrinth(file='lab_test.csv')
    num_won = 0
    gens = []
    fitnesses = []
    for i in range(num_tests):
        print(i)
        ga = GeneticAlgorithm(lab, num_population=100, max_iter=150, crossover_pts=1,
                          roulette_mult=2, max_moves_mult=2, file_name='try_{}'.format(i))
        ga.save_data()
        if ga.found_winner:
            num_won += 1
            gens.append(ga.winner_gen)
            fitnesses.append(ga.winner_fitness)
    print('Stats:\n№ won:\t{}\n№ gens req (avg):\t{}\nAvg fitness:\t{}' \
          .format(num_won, np.mean(gens), np.mean(fitnesses)))
if __name__ == '__main__':
    # testing stuff
    # logging.basicConfig(level=logging.DEBUG)
    # np.set_printoptions(threshold=np.nan)
    # ga.save_data()
    # ga.print_result()
    # GeneticAlgorithm.visualize_load('last_run')
    run(50)