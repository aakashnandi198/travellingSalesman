from collections import namedtuple

import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import numpy as np

MAX_ITERATIONS = 10**6
MAX_SEEDING_TRIALS = 10**5

class Solver:
    def __init__(self, points: namedtuple):
        self.points = points
        self.node_count = len(points)
        self.tabu={}
        self.tabu_limit = 2

    def length(self, point1: namedtuple, point2: namedtuple) -> float:
        return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5
    
    def objective(self, solution: list) -> float:
        obj = self.length(self.points[solution[-1]], self.points[solution[0]])
        for index in range(0, self.node_count - 1):
            obj += self.length(self.points[solution[index]], self.points[solution[index + 1]])
        return obj

    # show the solution on a graphical interface
    def visualize(self, solution: list):
        objective_value = self.objective(solution)
        x = [self.points[i].x for i in solution] + [self.points[solution[0]].x]
        y = [self.points[i].y for i in solution] + [self.points[solution[0]].y]
        plt.plot(x, y, 'o-')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('TSP Solution Visualization')
        plt.suptitle(f'Objective Value: {objective_value:.2f}', fontsize=10)
        plt.show()
        
    def greedy_solution(self) -> list:
        unvisited = set(range(self.node_count))
        current_node = 0
        solution = [current_node]
        unvisited.remove(current_node)

        while unvisited:
            next_node = min(unvisited, key=lambda node: self.length(self.points[current_node], self.points[node]))
            solution.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node
        return solution
    
    def select_seed_node(self, current_solution: list, itr : int) -> int:
        # Randome select a node as seed for k-opt that is not in tabu list
        candidates = [
            node for node in current_solution
            if node not in self.tabu or itr - self.tabu[node] > self.tabu_limit
        ]
        selected_node = random.choice(candidates)
        self.tabu[selected_node] = itr
        return selected_node
        
    
    def select_candidate(self, current_solution: list, point: int, itr :int, level:int) -> int:
        # Select a candidate node for k-opt swap
        # todo : improve selection strategy
        idx = current_solution.index(point)
        # the next idx is the current child, previous idx is current parent
        curr_child = current_solution[idx+1] if (idx+1)<len(current_solution) else current_solution[0]
        curr_parent = current_solution[idx-1]
        candidate_child = -1
        candidates = [
            node for node in current_solution
            if node != point and node != curr_child and node != curr_parent and (node not in self.tabu or itr - self.tabu[node] > self.tabu_limit)
        ]
        if not candidates:
            candidate_child = -1
        else:
            distances = [self.length(self.points[point], self.points[node]) for node in candidates]
            # Make shorter distances more likely as iterations increase
            # Use a softmax-like probability, with a temperature that decreases over iterations
            # Higher temperature = more random, lower = more greedy
            temperature = max(0.1, 10.0 / float(1 + level / 10000))
            inv_distances = np.exp(-np.array(distances) / temperature)
            probabilities = inv_distances / inv_distances.sum()
            candidate_child = random.choices(candidates, weights=probabilities, k=1)[0]
        self.tabu[candidate_child] = itr
        return candidate_child

    def k_optimize(self, current_solution: list, selected_node: int , itr :int, level : int) -> list:
        self.tabu[selected_node] = itr
        # select a candidate node to swap with
        candidate = self.select_candidate(current_solution, selected_node, itr, level)
        if candidate == -1:
            return current_solution
        # rotated solution to start from point
        idx_selected_node = current_solution.index(selected_node)
        rotated_solution = current_solution[idx_selected_node:] + current_solution[:idx_selected_node]
        idx_candidate = rotated_solution.index(candidate)
        candidate_solution = rotated_solution[:1] + rotated_solution[idx_candidate:0:-1]+ rotated_solution[idx_candidate+1:]
        # rotate back to original starting point
        idx_start = candidate_solution.index(current_solution[0])
        candidate_solution = candidate_solution[idx_start:] + candidate_solution[:idx_start]
        
        # check if candidate solution can be improved
        candidate_solution = self.k_optimize(candidate_solution, candidate, itr, level+1)

        candidate_value = self.objective(candidate_solution)
        current_value = self.objective(current_solution)

        if candidate_value < current_value:
            return candidate_solution
        else:
            del self.tabu[candidate]
            #print("No improvement at level ", level)
            return current_solution
    
    def optimal_solution(self) -> list:
        best_solution = self.greedy_solution()
        best_solution_value = self.objective(best_solution)
        for itr in tqdm(range(MAX_SEEDING_TRIALS)):
            # chose a random index for k-opt
            selected_node = self.select_seed_node(best_solution, itr)
            improved_solution = self.k_optimize(best_solution, selected_node, itr, 0)
            improved_solution_value = self.objective(improved_solution)
            if improved_solution_value < best_solution_value:
                best_solution = improved_solution
                best_solution_value = improved_solution_value
            #self.visualize(improved_solution)
        return best_solution

    def solve(self) -> list:
        # Trivial solution: visit nodes in the order they appear
        solution = self.optimal_solution()
        return solution