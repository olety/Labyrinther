if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    import os, sys

    # get an absolute path to the directory that contains mypackage
    foo_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
    sys.path.append(os.path.normpath(os.path.join(foo_dir, '..', '..')))
    from genetic.genetic import Labyrinth, Moveset, GeneticAlgorithm

    lab = Labyrinth(file='../../example_labyrinths/10x10.csv')
    # lab.plot_arr(lab.array_closed)
    print('Starting GA')
    ga = GeneticAlgorithm(lab, file_name='123')
    ga.save_data(
        file_dir=os.path.join('test', '1'),
        pic_last_plot=True,
        gif_full_plot=True,
        gif_last_plot=True,
        dyn_avg_fit=True,
        dyn_last_fit=True)
    # i = 0
    # while not ga.found_winner:
    #     ga = GeneticAlgorithm(lab)
    #     i += 1
    #     print(i)
    # print(ga.all_pops)
    # print(ga.found_winner)

    # ga.analyze()
    # run(2)
else:
    from bokeh.embed import components
    from bokeh.plotting import figure
    from matplotlib import pyplot as plt
    from matplotlib import animation
    from .labyrinth import Labyrinth
    from .moveset import Moveset
    import numpy as np
    import math
    import random
    import logging
    import os

    plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'


    class GeneticAlgorithm:
        def __init__(self, labyrinth, **kwargs):
            self.setup = kwargs
            # Setting the variables
            if type(labyrinth) is Labyrinth:
                self.labyrinth = labyrinth
            else:
                self.labyrinth = Labyrinth(file_obj=labyrinth)

            # Kwargs: General
            self.file_name = kwargs.get('file_name', 'last_run')
            # self.save = kwargs.get('save', True)

            # Kwargs: Algorithm - related
            # print(kwargs)
            self.best_movesets = []
            self.roulette_mult = int(kwargs.get('roulette_mult', 1))
            self.crossover_rate = float(kwargs.get('crossover_rate', 0.8))
            self.crossover_pts = kwargs.get('crossover_pts', 'random')
            self.mutation_rate = float(kwargs.get('mutation_rate', 0.001))
            self.selection = kwargs.get('selection', 'roulette')
            self.elites_cnt = int(kwargs.get('elitism_num', 5))
            self.pop_cnt = int(kwargs.get('num_population', 100))
            self.min_moves_mult = float(kwargs.get('min_moves_mult', 0.5))
            self.max_moves_mult = float(kwargs.get('max_moves_mult', 3))
            self.max_iter = int(kwargs.get('max_iter', 100))
            self.avg_fitness = []
            if self.crossover_pts == 'random':
                self.crossover_pts = random.randint(1, 3)
            self.crossover_pts = int(self.crossover_pts)
            # Generating the starting population
            self._generate_initial_pop()

            # TODO: Move this into a separate start() function?

            # Doing selection
            if self.selection == 'roulette':
                self.found_winner, self.max_gen = self._roulette_selection()
            elif self.selection == 'random':
                self.found_winner, self.max_gen = self._random_search()
            else:
                raise Exception('GA - Wrong selection method: {}'.format(self.selection))

        def _generate_initial_pop(self):
            self.pop = np.array([[Moveset(random.randint(
                int(self.labyrinth.num_moves_max * self.min_moves_mult),
                int(self.labyrinth.num_moves_max * self.max_moves_mult))), 0]
                                 for _ in range(self.pop_cnt)])
            self.next_pop = np.array([[None, 0] for _ in range(self.pop_cnt)])

        def _generate_random_moveset(self):
            return Moveset(random.randint(
                int(self.labyrinth.num_moves_max * self.min_moves_mult),
                int(self.labyrinth.num_moves_max * self.max_moves_mult)))

        def _roulette_selection(self):
            self.prev_pop = []
            logging.info('Starting roulette selection')
            roulette = np.empty(self.pop_cnt * self.roulette_mult, dtype=object)
            logging.debug('Inited empty roulette with size {}'.format(roulette.shape))
            self.winner_moveset = None
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
                self.avg_fitness.append(np.mean(self.pop[:, 1]))
                self.best_movesets.append(self.pop[0, 0])
                if self.winner_moveset:
                    indices = np.where(self.pop[:, 0] == self.winner_moveset)
                    self.winner_fitness = self.pop[indices[0], 1]
                    return True, gen
                self.pop[:, 1] -= self.pop[:, 1].min() - 1
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

            self.pop = self.pop[np.flipud(self.pop[:, 1].argsort())]
            self.pop[:, 1] -= self.pop[:, 1].min() - 1
            self.avg_fitness.append(np.mean(self.pop[:, 1]))
            return False, gen + 1

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

        def save_data(self, **kwargs):
            # if to_html:  # Package's .image_url doesn't work atm :(
            #     output_file(target_dir + '{}.html'.format(self.file_name))
            #     s1 = figure(width=500, plot_height=500, title='Fitness of the last population')
            #     s1.line(np.arange(len(self.pop[:, 1])), self.pop[:, 1])
            #
            #     s2 = figure(width=500, height=500, title='Average fitness (before making it positive)')
            #     s2.line(np.arange(len(self.avg_fitness)), self.avg_fitness)
            #
            #     self.plot_best(show=False, savefig=True, file_name='plots/{}.png'.format(self.file_name))
            #     # pic_src = ColumnDataSource(dict(url=[photo_dir+'{}.png'.format(self.file_name)]))
            #
            #     s3 = figure(x_range=(0, 1), y_range=(0, 1), title='Labyrinth')
            #     s3.image_url(url=[plot_dir + '{}.png'], x=0, y=1, h=0.9, w=0.9)
            #     p = gridplot([[s3, s1], [None, s2]])
            #     save(p)
            # else:

            # static/1/arr
            # static/1/plots

            plot_dir = os.path.join(kwargs.get('file_dir', ''), kwargs.get('plot_dir', ''))
            os.makedirs(plot_dir, exist_ok=True)
            if kwargs.get('pic_last_plot', False):  # Picture of the best individual in the last pop
                self.plot_best(show=False, savefig=True, file_name=os.path.join(plot_dir, 'last.png'))
            if kwargs.get('gif_full_plot', False):  # step by step progression of the GA
                self.plot_best_gif_all(file_name=os.path.join(plot_dir, 'full.gif'))
            if kwargs.get('gif_last_plot', False):  # step by step vis. of the best individual in the last pop
                self.plot_best_gif(file_name=os.path.join(plot_dir, 'last.gif'))

            if kwargs.get('dyn_last_fit'):  # Last pop fitness plot
                fig = figure(width=450, plot_height=450,
                             title='Fitness of the last population (before adjusting)', responsive=True)
                fig.line(np.arange(len(self.pop[:, 1])), self.pop[:, 1])
                np.save(os.path.join(plot_dir, 'dyn_last_fit'), components(fig))

            if kwargs.get('dyn_avg_fit'):  # Average fitness plot
                fig = figure(width=450, height=450, title='Average fitness (before adjusting)', responsive=True)
                fig.line(np.arange(len(self.avg_fitness)), self.avg_fitness)
                np.save(os.path.join(plot_dir, 'dyn_avg_fit'), components(fig))

            arr_path = os.path.join(kwargs.get('file_dir', ''), kwargs.get('arr_dir', ''))
            os.makedirs(arr_path, exist_ok=True)
            np.save(os.path.join(arr_path, 'arr.npy'),
                    [self.max_gen, self.max_iter, self.best_moveset, self.selection, self.avg_fitness,
                     self.setup, self.found_winner])

        def plot_best(self, **kwargs):
            self.labyrinth.plot_moveset(self.best_moveset, **kwargs)

        def plot_best_gif_all(self, **kwargs):
            self.labyrinth.plot_anim(self.best_movesets)

        def plot_best_gif(self, **kwargs):
            pass

        def print_result(self):
            if self.found_winner:
                print('Found a winner moveset! It was in gen {}, fitness: {}, moveset:{}'
                      .format(self.max_gen, self.winner_fitness, self.winner_moveset))
            else:
                print('This algorithm didn\'t find a winner yet :(')

        def analyze(self):
            print('Started analysis:')
            print('Num generations = {}'.format(self.max_gen))
            if self.winner_moveset:
                self.plot_best()
            if self.pop.all():
                GeneticAlgorithm.plot_fitness_pop(self.pop)
            if self.avg_fitness:
                GeneticAlgorithm.plot_fitness_avg(self.avg_fitness)

        @property
        def best_moveset(self):
            if self.winner_moveset:
                return self.winner_moveset
            else:
                return self.pop[0, 0]

        @staticmethod
        def _fitness(distance, bumps, redundancy, moves, length):
            return 0.3 * moves + 0.5 * length - redundancy - 10 * distance - bumps

        @staticmethod
        def plot_fitness_pop(pop):
            f, ax = plt.subplots()
            ax.plot(np.arange(len(pop[:, 1])), pop[:, 1])
            plt.show(block=False)

        @staticmethod
        def plot_fitness_avg(fit):
            f, ax = plt.subplots()
            ax.plot(np.arange(len(fit)), fit)
            plt.show(block=False)

        @staticmethod
        def analyze_file(filename):
            saved = np.load(filename + '.npy')
            pop = saved[0]
            num_gen = saved[1]
            winner_moveset = saved[2]
            labyrinth = saved[3]
            fitness_avg = saved[4]
            print('Num generations = {}'.format(num_gen))
            labyrinth.plot_moveset(winner_moveset)
            GeneticAlgorithm.plot_fitness_pop(pop)
            GeneticAlgorithm.plot_fitness_avg(fitness_avg)


    def run(num_tests):
        lab = Labyrinth(file='5x5.csv')
        ga_won = 0
        gens = []
        fitness = []
        print('Running GA tests ({})'.format(num_tests))
        for i in range(num_tests):
            print('Test №{} - '.format(i), end='')
            ga = GeneticAlgorithm(lab, elitism_num=2, num_population=100, max_iter=150, crossover_pts=1,
                                  roulette_mult=2, max_moves_mult=2, file_name='try_{}'.format(i))
            ga.save_data()
            if ga.found_winner:
                print('Found winner')
                ga_won += 1
                gens.append(ga.max_gen)
                fitness.append(ga.pop)
            else:
                gens.append(0)
                fitness.append(ga.pop)
                print('Didn\'t find winner')
        fitness = np.array(fitness)
        print(fitness.shape)
        print('Stats:\n№ won: \t{}\n% won: \t{}\n№ gens req (avg): \t{}\nAvg fitness: \t{}'
              .format(ga_won, 100 * ga_won / num_tests,
                      np.mean(gens), np.mean(fitness[:, :, 1])))
        # num_subplots = 0
        # plt.autoscale = True
        # f, ax = plt.subplot(2, len(fitness))
        #
        # for i, pop in enumerate(fitness[:, :, 1]):
        #     plt.subplot()
        #     ax[0][i].plot(pop[:, 1])

        ax_gens = plt.subplot(211)
        ax_gens.set_ylim([math.ceil(min(gens) - 0.5 * (max(gens) - min(gens))),
                          math.ceil(min(gens) + 0.5 * (max(gens) - min(gens)))])
        plt.bar(np.arange(num_tests), width=0.35, height=gens, color='m')
        print('Running random search tests ({})'.format(num_tests))
        iterations = []
        rand_won = 0
        for i in range(num_tests):
            print('Test №{} - '.format(i), end='')
            ga = GeneticAlgorithm(lab, selection='random', min_moves_mult=0.5, max_moves_mult=2,
                                  file_name='rng_try_{}'.format(i))
            if ga.found_winner:
                print('Found winner')
                rand_won += 1
            else:
                print('Didn\'t find winner')
            iterations.append(ga.max_gen)
        print('Stats:\n№ won: \t{}\n% won: \t{}\n№ iters (avg): \t{}'
              .format(rand_won, 100 * rand_won / num_tests, np.mean(iterations)))

        ax_iterations = plt.subplot(212, sharex=ax_gens)
        ax_iterations.set_ylim([math.ceil(min(iterations) - 0.5 * (max(iterations) - min(iterations))),
                                math.ceil(min(iterations) + 0.5 * (max(iterations) - min(iterations)))])
        plt.bar(np.arange(num_tests), width=0.35, height=iterations, color='y')

        # ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))
        plt.show(block=False)
