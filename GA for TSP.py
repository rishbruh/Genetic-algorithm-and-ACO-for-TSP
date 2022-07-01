import pygame
import random
import csv
import math
from collections import Counter
import time

class SolveTSP:
    def __init__(self):
        self.visuals = True
        self.mutation_rate = 0.01
        pop_size = 1000

        with open('oliver30.csv', newline='') as csvfile:
            data = list(csv.reader(csvfile))

        for row in data:
            row[1] = float(row[1])
            row[0] = float(row[0])

        self.cities = [tuple(x) for x in data]

        print(self.cities)

        self.popln = [None] * pop_size
        l_population = len(self.popln)
        i = 0
        while i < l_population:
            random.shuffle(self.cities)
            self.popln[i] = self.cities[:]
            i+=1
        self.fit = [0] * pop_size

        self.present_best = {"length": float('inf'),"path": self.popln[0]}

        self.countofgen = 0

    def start(self):
        self.stamp = time.time()
        if self.visuals:
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 32)
            self.screen = pygame.display.set_mode((801, 851))
            size = (450, 850)
            self.surface = pygame.Surface(size)

            self.main_loop()
        else:
            while True:
                self.countofgen = 1 + self.countofgen
                self.calc_fit()
                self.generate_new_popln()

    def main_loop(self):
        rbg_max = 255
        current_generation_text = self.font.render("Initialising", False, (rbg_max, rbg_max, rbg_max))
        alltime_best_text = self.font.render("Initialising", False, (rbg_max, rbg_max, rbg_max))
        min_fn = (0, 0, 0)
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            self.surface.fill(min_fn)
            self.screen.fill(min_fn)

            self.draw_cities()

            self.countofgen += 1
            self.calc_fit()
            self.generate_new_popln()

            max_idx = self.fit.index(max(self.fit))
            current_gen_best = self.popln[max_idx]

            self.draw_path(current_gen_best, (rbg_max, rbg_max, rbg_max), 1, 0)
            self.draw_path(self.present_best["path"], (rbg_max, rbg_max, rbg_max), 1, 1)
            current_generation_text = self.font.render("Generation {0} best".format(self.countofgen), False,
                                                       (rbg_max, rbg_max, rbg_max))
            alltime_best_text = self.font.render("Current best: {0}".format(math.floor(self.present_best["length"])),
                                                 False, (rbg_max, rbg_max, rbg_max))

            self.screen.blit(self.surface, (101, 101))
            self.screen.blit(current_generation_text, (451, 101))
            self.screen.blit(alltime_best_text, (451, 501))

            pygame.display.update()

    def generate_new_popln(self):
        length_popln = len(self.popln)
        new_pop = [None] * length_popln

        i = 0
        while i < length_popln:
            parentA, parentB = None, None
            iter_limit = 0
            while parentA == parentB:
                parentB = self.select_GA()
                parentA = self.select_GA()
                iter_limit += 1
                if iter_limit > 50000:
                    break
            new_pop[i] = self.crossover(parentB, parentA)
            new_pop[i] = self.mutate(new_pop[i])
            i+=1

        self.popln = new_pop

    def select_GA(self):
        max_fit = max(self.fit)

        l_population = len(self.popln)-1
        idx = random.randint(0, l_population)

        while self.fit[idx] < random.uniform(0, max_fit):
            idx = random.randint(0, l_population)
        return self.popln[idx]

    def crossover(self, pB, pA):
        rand = random.randint(0, len(pA) - 1)
        offspring = pA[:rand]
        intersect = Counter(offspring) & Counter(pB)
        pB = (Counter(pB) - intersect).elements()
        pB = list(pB)
        offspring = offspring + pB[:]
        return offspring

    def mutate(self, p):
        l_p = len(p)
        i=0
        while i < l_p:
            if random.random() < self.mutation_rate:
                l_cities = len(self.cities) - 1
                idx = random.randint(0, l_cities)
                p[i], p[idx] = p[idx], p[i]
                i+=1
        return p

    def calc_dist(self, p):
        total = 0
        length_p = len(p) - 1
        i = 0
        while i < length_p:
            x = (p[i + 1][0] - p[i][0]) ** 2
            y = (p[i + 1][1] - p[i][1]) ** 2
            total += pow(x + y, 1 / 2)
            i += 1

        x = (p[-1][0] - p[0][0]) ** 2
        y = (p[-1][1] - p[0][1]) ** 2
        total = total + math.sqrt(x + y)
        return total

    def calc_fit(self):
        total_fit = 0
        length_ga = len(self.fit)
        i = 0
        while i < length_ga:
            self.fit[i] = self.calc_dist(self.popln[i])
            time_var = time.time() - self.stamp
            if (self.fit[i] < self.present_best["length"]):
                print("New best path is found: {0} after {1} generations and {2:5.2f} seconds.".format(self.fit[i],
                                                                                                    self.countofgen,
                                                                                                    time_var))
                self.present_best["path"] = self.popln[i]
                self.present_best["length"] = self.fit[i]
            self.fit[i] = 1 / self.fit[i]
            total_fit = total_fit + self.fit[i]
            i+=1



    def draw_cities(self):
        length_cities = len(self.cities)
        i = 0
        rbg_max=255
        while i < length_cities:
            pygame.draw.circle(self.surface, (rbg_max, rbg_max, rbg_max), (self.cities[i][0], self.cities[i][1]), 3)
            pygame.draw.circle(self.surface, (rbg_max, rbg_max, rbg_max), (self.cities[i][0], self.cities[i][1] + 400), 3)
            i+=1

    def draw_path(self, p, colour, width, y_multi):
        offset = 400 * y_multi
        len_p = len(p) - 1
        i = 0
        while i < len_p:
            pygame.draw.line(self.surface, colour, (p[i][0], p[i][1] + offset), (p[i + 1][0], p[i + 1][1] + offset),
                             width)
            i+=1
        pygame.draw.line(self.surface, colour, (p[-1][0], p[-1][1] + offset), (p[0][0], p[0][1] + offset), width)


tsp = SolveTSP()
tsp.start()