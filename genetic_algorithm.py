from labyrinth import Labyrinth
from moveset import Moveset
import numpy as np
import math
import scipy.special as scp
from matplotlib import pyplot as plt
import random
import logging


class GeneticAlgorithm:
    def __init__(self, labyrinth, **kwargs):
        # Setting the variables
        self.labyrinth = labyrinth
        # Kwargs: General
        self.file_name = kwargs.get('file_name', 'last_run')
        self.save = kwargs.get('save', True)
        # Kwargs: Algorithm - related
        self.roulette_mult = kwargs.get('roulette_mult', 1)
        self.crossover_rate = kwargs.get('crossover_rate', 0.8)
        self.crossover_pts = kwargs.get('crossover_pts', 'random')
        self.mutation_rate = kwargs.get('mutation_rate', 0.001)
        self.selection = kwargs.get('selection', 'roulette')
        self.elites_cnt = kwargs.get('elitism_num', 5)
        self.pop_cnt = kwargs.get('num_population', 100)
        self.min_moves_mult = kwargs.get('min_moves_mult', 0.5)
        self.max_moves_mult = kwargs.get('max_moves_mult', 3)
        self.max_iter = kwargs.get('max_iter', 100)

        if self.crossover_pts == 'random':
            self.crossover_pts = random.randint(1, 3)

        # Generating the starting population
        self._generate_initial_pop()

        # Doing selection
        if self.selection == 'roulette':
            self.found_winner, self.max_gen = self._roulette_selection()
        elif self.selection == 'random':
            self.found_winner, self.max_gen = self._random_search()
        else:
            raise Exception('GA - Wrong selection method: {}'.format(self.selection))

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
                    self.winner_moveset = person[0]
            # Ensuring that fitness is >= 1
            logging.debug('Starting initial population fitness eval')
            logging.info('Sorting the population')
            self.pop = self.pop[np.flipud(self.pop[:, 1].argsort())]
            self.pop[:, 1] -= self.pop[:, 1].min() - 1
            if self.winner_moveset:
                indices = np.find(self.pop[:, 0] == self.winner_moveset)
                self.winner_fitness = self.pop[indices[0], 1]
                return True, gen
            fitness_sum = self.pop[:, 1].sum()
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
            self.pop = self.next_pop
        return False, gen + 1

    def print_result(self):
        if self.found_winner:
            print('Found a winner moveset! It was in gen {}, fitness: {}, moveset:{}'
                  .format(self.max_gen, self.winner_fitness, self.winner_moveset))
        else:
            print('This algorithm didn\'t find a winner yet :(')

    def save_data(self):
        np.save(self.file_name, [self.pop, self.max_gen, self.winner_moveset, self.labyrinth, self.selection])

    @staticmethod
    def plot_fitness(pop):
        f, ax = plt.subplots()
        ax.scatter([range(len(pop))], pop[:, 1])
        plt.show()

    @staticmethod
    def analyze(filename):
        saved = np.load(filename + '.npy')
        pop = saved[0]
        num_gen = saved[1]
        winner_moveset = saved[2]
        labyrinth = saved[3]
        print('Num generations = {}'.format(num_gen))
        labyrinth.plot_moveset(winner_moveset)
        GeneticAlgorithm.plot_fitness(pop)

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
            self.labyrinth.plot_moveset(self.winner_moveset)  # , savefig=True, file_name='labyrinth.png')
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
