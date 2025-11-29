#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
from travellingsalesman.solver import Solver
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio

Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def visualize(points, solution: list, solution_value:  int,  filename: str, window_name: str = "TSP Solution Visualization") -> None:
    # Matplotlib static plot
    plt.figure(window_name)
    plt.clf()
    objective_value = solution_value
    x = [points[i].x for i in solution] + [points[solution[0]].x]
    y = [points[i].y for i in solution] + [points[solution[0]].y]
    plt.plot(x, y, 'o-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('window_name')
    plt.suptitle(f'Objective Value: {objective_value:.2f}', fontsize=10)
    plt.savefig("./ans/" + filename+".png")
    plt.close()

    # Plotly interactive plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Tour'))
    fig.update_layout(
        title=f"{window_name}",
        xaxis_title='X',
        yaxis_title='Y',
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
    )
    fig.update_layout(
        title_text=f"Objective Value: {objective_value:.2f}",
        title_font_size=18
    )
    # Save as interactive HTML
    pio.write_html(fig, f"./ans/{filename}.html", auto_open=False)

def record_solution(points, solution, solution_value,filename):
    # render the solution to ans/ directory
    visualize(points, solution, solution_value, filename)

    with open("./ans/"+filename, "w") as f:
        f.write('%.2f' % solution_value + ' ' + str(0) + '\n')
        f.write(' '.join(map(str, solution)) + '\n')

def solve_it(input_data, filename):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    solver = Solver(points)
    solution = solver.solve()
    solution_value = solver.objective(solution)
    record_solution(points, solution, solution_value, filename)
    #solver.visualize(solution)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        # extract the filename without path and extension
        filename = file_location.split('/')[-1]
        print(solve_it(input_data, filename))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

