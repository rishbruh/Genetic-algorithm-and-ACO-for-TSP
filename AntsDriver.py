import csv
import numpy as np
from AntsOG import *
from time import sleep

with open('oliver30.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
for row in data:
    row[1] = float(row[1])
    row[0] = float(row[0])

location = [tuple(x) for x in data]


if __name__ == '__main__':


    _colony_size = 50
    _steps = 50
    _nodes = location

    acs = ACOforTSP(mode='ACS', colony_size=_colony_size, steps=_steps, nodes=_nodes)
    time, dist = acs.run()
    route = acs.global_best_tour
    print('Runtime: ', time, 's')
    print('Minimum distance: ', dist)
    acs.plot()