from collections import namedtuple

import matplotlib.pyplot as plt
import random
import tqdm

MAX_ITERATIONS = 10

class Solver:
    def __init__(self, points: namedtuple):
        self.points = points
        self.node_count = len(points)

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
    
    def select_seed_node(self, current_solution: list) -> int:
        # Select a random node from the current solution as seed for k-opt
        return random.choice(current_solution)
    
    def select_candidate(self, current_solution: list, point: int) -> int:
        # Select a candidate node for k-opt swap
        # todo : improve selection strategy
        idx = current_solution.index(point)
        curr_child = current_solution[idx+1]
        curr_parent = current_solution[idx-1]
        candidate_child = -1
        min_distance = float('inf')
        for node in current_solution:
            if node != point and node != curr_child and node != curr_parent:
                dist = self.length(self.points[point], self.points[node])
                if dist < min_distance:
                    min_distance = dist
                    candidate_child = node
        return candidate_child

    def k_optimize(self, current_solution: list, selected_node: int) -> list:
        # Perform k-opt swap at index idx
        candidate = self.select_candidate(current_solution, selected_node)
        # rotated solution to start from point
        idx_selected_node = current_solution.index(selected_node)
        rotated_solution = current_solution[idx_selected_node:] + current_solution[:idx_selected_node]
        idx_candidate = rotated_solution.index(candidate)
        improved_solution = rotated_solution[:1] + rotated_solution[idx_candidate:0:-1]+ rotated_solution[idx_candidate+1:]
        # rotate back to original starting point
        idx_start = improved_solution.index(current_solution[0])
        improved_solution = improved_solution[idx_start:] + improved_solution[:idx_start]
        return improved_solution
    
    def optimal_solution(self) -> list:
        best_solution = self.greedy_solution()
        best_solution_value = self.objective(best_solution)
        for itr in range(MAX_ITERATIONS):
            # chose a random index for k-opt
            selected_node = self.select_seed_node(best_solution)
            improved_solution = self.k_optimize(best_solution, selected_node)
            improved_solution_value = self.objective(improved_solution)
            if improved_solution_value < best_solution_value:
                best_solution = improved_solution
                best_solution_value = improved_solution_value
            self.visualize(improved_solution)
        return best_solution

    def solve(self) -> list:
        # Trivial solution: visit nodes in the order they appear
        solution = self.optimal_solution()
        return solution