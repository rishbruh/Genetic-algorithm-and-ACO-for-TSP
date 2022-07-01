import math
import random
import time as tm
from tqdm import tqdm
from matplotlib import pyplot as plt

class ACOforTSP:
    class Edge:
        def __init__(self, a, b, wt, start_pheromone):
            self.a = a
            self.b = b
            if wt == 0:
                wt = 1e-10
            self.wt = wt
            self.pheromone = start_pheromone

    class Ant:
        def __init__(self, alpha, beta, nodesnum, edges):
            self.alpha = alpha
            self.beta = beta
            self.nodesnum = nodesnum
            self.edges = edges
            self.tour = None
            self.distance = 0.0

        def choose_node(self):
            roulette_wheel = 0.0
            unvisited_nodes = [node for node in range(self.nodesnum) if node not in self.tour]
            heuristic_total = 0.0
            for unvisited_node in unvisited_nodes:
                heuristic_total += self.edges[self.tour[-1]][unvisited_node].wt
            for unvisited_node in unvisited_nodes:
                roulette_wheel += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].wt), self.beta)
            random_value = random.uniform(0.0, roulette_wheel)
            wheel_position = 0.0
            for unvisited_node in unvisited_nodes:
                wheel_position += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].wt), self.beta)
                if wheel_position >= random_value:
                    return unvisited_node

        def tourfinder(self):
            self.tour = [random.randint(0, self.nodesnum - 1)]
            while len(self.tour) < self.nodesnum:
                self.tour.append(self.choose_node())
            return self.tour

        def get_distance(self):
            self.distance = 0.0
            for i in range(self.nodesnum):
                self.distance += self.edges[self.tour[i]][self.tour[(i + 1) % self.nodesnum]].wt
            return self.distance

    def __init__(self, mode='ACS', colony_size=10,alpha=1.0, beta=3.0,
                 rho=0.3, pheromone_wt=1.0, start_pheromone=1.0, steps=100, nodes=None, labels=None):
        self.mode = mode
        self.colony_size = colony_size
        self.rho = rho
        self.pheromone_wt = pheromone_wt
        self.steps = steps
        self.nodesnum = len(nodes)
        self.nodes = nodes
        if labels is not None:
            self.labels = labels
        else:
            self.labels = range(1, self.nodesnum + 1)
        self.edges = [[None] * self.nodesnum for _ in range(self.nodesnum)]
        for i in range(self.nodesnum):
            for j in range(i + 1, self.nodesnum):
                self.edges[i][j] = self.edges[j][i] = self.Edge(i, j, math.sqrt(
                    pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0)),
                                                                start_pheromone)
        self.ants = [self.Ant(alpha, beta, self.nodesnum, self.edges) for _ in range(self.colony_size)]
        self.global_best_tour = None
        self.global_best_distance = float("inf")

    def addingpheromone(self, tour, distance, wt=1.0):
        pheromone_to_add = self.pheromone_wt / distance
        for i in range(self.nodesnum):
            self.edges[tour[i]][tour[(i + 1) % self.nodesnum]].pheromone += wt * pheromone_to_add

    def _acs(self):
        for step in range(self.steps):
            for ant in self.ants:
                self.addingpheromone(ant.tourfinder(), ant.get_distance())
                if ant.distance < self.global_best_distance:
                    self.global_best_tour = ant.tour
                    self.global_best_distance = ant.distance
            for i in range(self.nodesnum):
                for j in range(i + 1, self.nodesnum):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)

    def run(self):
        start = tm.time()
        if self.mode == 'ACS':
            self._acs()


        runtime = tm.time() - start
        return runtime, self.global_best_distance

    def plot(self, line_width=1, point_radius=math.sqrt(2.0), annotation_size=8, dpi=120, save=True, name=None):
        x = [self.nodes[i][0] for i in self.global_best_tour]
        x.append(x[0])
        y = [self.nodes[i][1] for i in self.global_best_tour]
        y.append(y[0])
        plt.plot(x, y, linewidth=line_width)
        plt.scatter(x, y, s=math.pi * (point_radius ** 2.0))
        plt.title(self.mode)
        for i in self.global_best_tour:
            plt.annotate(self.labels[i], self.nodes[i], size=annotation_size)

        plt.show()


