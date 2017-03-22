from labyrinth import Labyrinth
from moveset import Moveset
import numpy as np
import math
import scipy.special as scp
from matplotlib import pyplot as plt
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
            self.found_winner, self.max_gen = self._roulette_selection()
        elif selection == 'random':
            self.found_winner, self.max_gen = self._random_search()
        else:
            raise Exception('GA - Wrong selection method: {}'.format(selection))

    def _generate_initial_pop(self):
        self._generate_random_pop()
        self.next_pop = np.array([[None, 0] for _ in range(self.pop_cnt)])
        # self.next_pop = self.pop.copy()
        # self.all_pops = self.pop.copy()

    def _generate_random_pop(self):
        self.pop = np.array([[Moveset(random.randint(
            int(self.labyrinth.num_moves_max * self.min_moves_mult),
            int(self.labyrinth.num_moves_max * self.max_moves_mult))), 0]
                             for _ in range(self.pop_cnt)])

    @staticmethod
    def _fitness(distance, bumps, redundancy, moves, length):
        # Using sigmoid (expit) function to eliminate negative vals
        return 0.3 * moves + 0.5 * length - redundancy - 10 * distance - bumps

    def _roulette_selection(self):
        self.prev_pop = []
        logging.info('Starting roulette selection')
        roulette = np.empty(self.pop_cnt * self.roulette_mult, dtype=object)
        logging.debug('Inited empty roulette with size {}'.format(roulette.shape))
        self.found_winner = False
        for gen in range(self.max_iter):
            logging.info('Generation №{}'.format(gen))
            # Rank the population
            logging.debug('Processing generation №{}'.format(gen))
            for i, person in enumerate(self.pop):
                distance, bumps, redundancy, moves = self.labyrinth.process_moveset(person[0])
                person[1] = GeneticAlgorithm._fitness(distance, bumps, redundancy, moves, len(person[0]))
                if distance == 0:
                    logging.info('GA - gen {} - Acquired a winner - moveset {}'.format(gen, person[0]))
                    self.found_winner = True
                    self.winner_moveset = person[0]
                    self.pop[:, 1] += (self.pop[:, 1] - self.pop[:, 1].min()) / self.pop[:, 1].ptp(0)
                    self.winner_fitness = 1
                    # self.all_pops = np.vstack((self.all_pops, self.pop))
                    return True, gen
            logging.debug('Starting initial population fitness eval')
            logging.info('Sorting the population')
            # Sorting the pop and adjusting the fitness to be always >= 1
            self.pop = self.pop[np.flipud(self.pop[:, 1].argsort())]
            self.pop[:, 1] -= self.pop[:, 1].min() - 1
            fitness_sum = self.pop[:, 1].sum()

            # ptp = self.pop[:, 1].ptp()
            # if ptp == 0:
            #     # print('Converged to a local minimum.')
            #     ptp = 0.1
            # self.pop[:, 1] = 100*(self.pop[:, 1] - self.pop[:, 1].min())/ptp

            # sum_fitness = np.sum(self.pop[:, 1])
            # Logging stuff
            # logging.info('Gen {} fitness sum = {}'.format(gen, sum_fitness))
            # logging.info('Gen {} fitness avg = {}'.format(gen, np.mean(self.pop[:, 1])))

            # Determine best ones - Make the roulette
            logging.info('Filling in the roulette')
            curr_pt = 0
            for person in self.pop:
                part = math.ceil((person[1] / fitness_sum) * len(roulette))
                if curr_pt + part >= len(roulette):
                    part = len(roulette) - curr_pt
                roulette[curr_pt:curr_pt + part] = np.repeat(person[0], part)
                curr_pt += part
            logging.debug('Filled roulette: {}'.format(roulette))

            # Elitism
            logging.info('Filling in the elites')
            for i in range(self.elites_cnt):
                self.next_pop[i][0] = self.pop[i][0]

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
            # self.all_pops = np.vstack((self.all_pops, self.pop))
            self.prev_pop = self.pop
            self.pop = self.next_pop
            # plt.plot(self.pop[:, 1])
            # plt.show()
        return False, gen + 1

    def print_result(self):
        if self.found_winner:
            print('Found a winner moveset! It was in gen {}, fitness: {}, moveset:{}'
                  .format(self.max_gen, self.winner_fitness, self.winner_moveset))
        else:
            print('This algorithm didn\'t find a winner yet :(')

    def save_data(self):
        # if self.all_pops.size > 0:
        #     if self.found_winner:
        #         self.all_pops = self.all_pops[self.pop_cnt:].reshape((self.max_gen + 1, self.pop_cnt, 2))
        #     else:
        #         self.all_pops = self.all_pops[self.pop_cnt:].reshape((self.max_gen, self.pop_cnt, 2))
        #     np.save(self.file_name, self.all_pops)
        # else:
        #     np.save(self.file_name, [self.pop, self.max_gen])
        pass

    # @staticmethod
    # def analyze(filename):
    #     all_pops = np.load(filename + '.npy')
    #     max = -100000000
    #     max_person = None
    #     for i, gen in enumerate(all_pops):
    #         for j, person in enumerate(gen):
    #             if person[1] >= max:
    #                 max = person[1]
    #                 max_person = person[0]
    #     # print(all_pops)
    #     print(max, max_person)

    def _generate_random_moveset(self):
        return Moveset(random.randint(
            int(self.labyrinth.num_moves_max * self.min_moves_mult),
            int(self.labyrinth.num_moves_max * self.max_moves_mult)))

    def _random_search(self):
        found = False
        i = 0
        while not found or i > 100000:
            mvs = self._generate_random_moveset()
            if self.labyrinth.process_moveset(mvs)[0] == 0:
                found = True
            else:
                i += 1
        return found, i

    def plot_best(self):
        if self.found_winner:
            self.labyrinth.plot_moveset(self.winner_moveset)
        elif self.all_pops.size > 0:
            print(self.all_pops.shape)
            self.labyrinth.plot_moveset(self.all_pops[self.all_pops.shape[0] - 1, 0])
            # plt.show()


def run(num_tests):
    lab = Labyrinth(file='lab_test.csv')
    num_won = 0
    gens = []
    fitnesses = []
    all_last_fitnesses = []
    print('Running GA tests ({})'.format(num_tests))
    for i in range(num_tests):
        print('Test №{} - '.format(i), end='')
        ga = GeneticAlgorithm(lab, elitism_num=2, num_population=100, max_iter=150, crossover_pts=1,
                              roulette_mult=2, max_moves_mult=2, file_name='try_{}'.format(i))
        ga.save_data()
        all_last_fitnesses.append(ga.prev_pop[:, 1])
        if ga.found_winner:
            print('Found winner')
            num_won += 1
            gens.append(ga.max_gen)
            fitnesses.append(ga.winner_fitness)
        else:
            gens.append(0)
            fitnesses.append(0)
            print('Didn\'t find winner')
    print('Stats:\n№ won: \t{}\n% won: \t{}\n№ gens req (avg): \t{}\nAvg fitness: \t{}'
          .format(num_won, 100 * num_won / num_tests,
                  np.mean(gens), np.mean(fitnesses)))
    for fitness in all_last_fitnesses:
        plt.figure()
        plt.plot(fitness)
    plt.figure()
    plt.plot(gens)
    print('Running random search tests ({})'.format(num_tests))
    iters = []
    for i in range(1):
        print('Test №{} - '.format(i), end='')
        ga = GeneticAlgorithm(lab, selection='random', min_moves_mult=0.5, max_moves_mult=2,
                              file_name='rng_try_{}'.format(i))
        if ga.found_winner:
            print('Found winner')
            num_won += 1
        else:
            print('Didn\'t find winner')
        iters.append(ga.max_gen)
    print('Stats:\n№ won: \t{}\n% won: \t{}\n№ iters (avg): \t{}'
          .format(num_won, 100 * num_won / num_tests, np.mean(iters)))
    plt.figure()
    plt.plot(iters)
    plt.show()

if __name__ == '__main__':
    lab = Labyrinth(file='lab_test.csv')
    print('Starting GA')
    ga = GeneticAlgorithm(lab)
    # print(ga.all_pops)
    print(ga.found_winner)
    if ga.found_winner:
        ga.plot_best()
    # run(5)
