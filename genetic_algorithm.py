from labyrinth import Labyrinth
from moveset import Moveset


class GeneticAlgorithm():
    def __init__(self, num_population, max_iter, labyrinth):
        print(labyrinth)


if __name__ == '__main__':
    lab = Labyrinth(file='lab_test.csv')
    ga = GeneticAlgorithm(100, 100, lab)
