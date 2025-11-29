from collections import namedtuple

import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import numpy as np

#MAX_SEEDING_TRIALS = 1
#MAX_TRIALS = 1000

class Solver:
    def __init__(self, points: namedtuple):
        self.points = points
        self.node_count = len(points)
        self.node_mem = []
        self.max_depth = 5
        self.max_branch =10
        self.tabu={}
        self.tabu_limit= 1 #max(2, int(self.node_count**0.33))
        self.max_seed_trials = 10
        self.max_trials = 1000

    def length(self, point1: namedtuple, point2: namedtuple) -> float:
        return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5
    
    def objective(self, solution: list) -> float:
        obj = self.length(self.points[solution[-1]], self.points[solution[0]])
        for index in range(0, self.node_count - 1):
            obj += self.length(self.points[solution[index]], self.points[solution[index + 1]])
        return obj

    # show the solution on a graphical interface
    def visualize(self, solution: list, window_name: str = "TSP Solution Visualization") -> None:
        # spawn a window with window_name if it does not exist else clear the window and render
        plt.figure(window_name)
        plt.clf()
        objective_value = self.objective(solution)
        x = [self.points[i].x for i in solution] + [self.points[solution[0]].x]
        y = [self.points[i].y for i in solution] + [self.points[solution[0]].y]
        plt.plot(x, y, 'o-')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('window_name')
        plt.suptitle(f'Objective Value: {objective_value:.2f}', fontsize=10)
        # donot pause execution, just render
        plt.pause(0.1)
        plt.draw()
        
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
    
    # select seed node that have nodes nearby and yet connected to far nodes
    def select_seed_node(self, current_solution: list, itr : int) -> int:
        # Build list of eligible nodes (respecting tabu)
        candidates = [node for node in current_solution if (node not in self.tabu or itr - self.tabu[node] > self.tabu_limit)]
        if not candidates:
            return -1

        scores = []
        # For each candidate compute how much it could benefit from reconnecting to a nearby non-adjacent node
        for node in candidates:
            idx = current_solution.index(node)
            prev_node = current_solution[idx-1]
            next_node = current_solution[idx+1] if (idx+1) < len(current_solution) else current_solution[0]

            # incident edge lengths
            d_prev = self.length(self.points[node], self.points[prev_node])
            d_next = self.length(self.points[node], self.points[next_node])
            longest_incident = max(d_prev, d_next)

            # find nearest non-adjacent node in the tour (a potential better connection)
            others = [n for n in current_solution if n not in (node, prev_node, next_node)]
            if not others:
                scores.append(0.0)
                continue
            dists = [self.length(self.points[node], self.points[n]) for n in others]
            nearest = min(dists)

            # potential reduction in edge length if we could reconnect the long incident edge to the nearby node
            reduction = max(0.0, longest_incident - nearest)
            #percentage_reduction = reduction / longest_incident if longest_incident > 0 else 0.0

            # score favors larger absolute reductions and also larger current incident lengths
            # add small epsilon to avoid zero weights
            score = reduction * (1.0 + longest_incident) + 1e-8
            #score = percentage_reduction +1e-8
            # penalize nodes present in node_mem (recently used in same k-opt)
            if node in self.node_mem:
                score *= 0.3
            scores.append(score)

        total = sum(scores)
        if total <= 0:
            # fallback: random eligible node
            return random.choice(candidates)

        # weighted random pick according to score
        selected = random.choices(candidates, weights=scores, k=1)[0]
        return selected

    def select_candidate(self, current_solution: list, point: int, explored_candidates: list[int] ,itr: int, level:int) -> int:
        # Select a candidate node for k-opt swap using prioritized logic
        idx = current_solution.index(point)
        curr_child = current_solution[idx+1] if (idx+1)<len(current_solution) else current_solution[0]
        curr_parent = current_solution[idx-1]
        node_count = len(current_solution)
        candidate_child = -1
        candidates = [
            node for node in current_solution
            if node != point and node != curr_child and node != curr_parent and (node not in self.node_mem) and (node not in explored_candidates) and (node not in self.tabu or itr - self.tabu[node] > self.tabu_limit)
        ]
        if not candidates:
            return -1

        scores = []
        # For each candidate, score by potential reduction in longest incident edge
        for node in candidates:
            # If we swapped point's connection to candidate, what would be the reduction?
            d_parent = self.length(self.points[point], self.points[curr_parent])
            d_child = self.length(self.points[point], self.points[curr_child])
            longest_incident = max(d_parent, d_child)
            d_candidate = self.length(self.points[point], self.points[node])
            reduction = max(0.0, longest_incident - d_candidate)
            #percentage_reduction = reduction / longest_incident if longest_incident > 0 else 0.0
            score = reduction * (1.0 + longest_incident) + 1e-8
            #score = percentage_reduction + 1e-8
            if node in self.node_mem:
                score *= 0.3
            scores.append(score)

        total = sum(scores)
        if total <= 0:
            # fallback: random eligible candidate
            return random.choice(candidates)

        candidate_child = random.choices(candidates, weights=scores, k=1)[0]
        return candidate_child

    def k_optimize(self, current_solution: list, selected_node: int , itr ,level : int) -> list:
        # print(f'K-opt level {level}, selected node {selected_node}')
        # print self.node_mem
        #print(f'nodes_in_memory: {self.node_mem}')
        # set current value and best value
        best_solution = current_solution
        best_value = self.objective(current_solution)
        
        # if maximum depth reached, return current solution
        if level >= self.max_depth:
            return current_solution
        
        # add selected node to memory to avoid re-selection in this k-opt
        self.node_mem.append(selected_node)
        
        explored_candidates = []
        # select a candidate node to swap with
        # reduce the branch factor as level increases to control combinatorial explosion
        for branch in range(self.max_branch // (2**(level-1))):
            candidate = self.select_candidate(current_solution, selected_node, explored_candidates, itr ,level)
            if candidate == -1:
                break
            
            # add candidate to explored list
            explored_candidates.append(candidate)
            # add candidate to memory to avoid re-selection in this k-opt
            #self.node_mem.append(candidate)

            # if there is tabu value for candidate store for reassigning later
            # prev_tabu = self.tabu.get(candidate, None)
            # set tabu for candidate
            # self.tabu[candidate] = itr
            
            # rotated solution to start from point
            idx_selected_node = current_solution.index(selected_node)
            rotated_solution = current_solution[idx_selected_node:] + current_solution[:idx_selected_node]
            idx_candidate = rotated_solution.index(candidate)
            
            # generate candidate solution by reversing the segment between selected_node and candidate
            candidate_solution = rotated_solution[:1] + rotated_solution[idx_candidate:0:-1]+ rotated_solution[idx_candidate+1:]
            
            # rotate back to original starting point
            idx_start = candidate_solution.index(current_solution[0])
            candidate_solution = candidate_solution[idx_start:] + candidate_solution[:idx_start]
        
            # check if candidate solution can be improved
            candidate_solution = self.k_optimize(candidate_solution, candidate, itr, level+1)
            candidate_value = self.objective(candidate_solution)
        
            if candidate_value < best_value:
                best_solution = candidate_solution
                best_value = candidate_value

            # remove candidate from memory after processing
            #self.node_mem.remove(candidate)
            # restore previous tabu value for candidate
            #if prev_tabu is not None:
            #    self.tabu[candidate] = prev_tabu

        # remove selected node from memory before returning    
        self.node_mem.remove(selected_node)
        # return the best solution found
        return best_solution
    
    def optimal_solution(self) -> list:
        greedy_solution = self.greedy_solution()
        greey_solution_value = self.objective(greedy_solution)

        ultra_solution = greedy_solution
        ultra_solution_value = greey_solution_value

        # setup dictionary to track improvement
        improvement_value_tracker = {}
        for seeding in range(self.max_seed_trials):
            best_solution = greedy_solution
            best_solution_value = greey_solution_value

            #reset tabu list
            self.tabu = {}
            for itr in tqdm(range(self.max_trials)):
                # chose a random index for k-opt
                selected_node = self.select_seed_node(best_solution, itr)
                if selected_node == -1:
                    continue
                improved_solution = self.k_optimize(best_solution, selected_node, itr, 1)
                improved_solution_value = self.objective(improved_solution)
                if improved_solution_value < best_solution_value:
                    best_solution = improved_solution
                    best_solution_value = improved_solution_value
                    # set tabu for nodes in best solution
                    for node in best_solution:
                        self.tabu[node] = itr
                    improvement_value_tracker[itr] = best_solution_value
                #self.visualize(improved_solution,"improved solution")
            # plot the improvement over iterations
            #plt.plot(list(improvement_value_tracker.keys()), list(improvement_value_tracker.values()))
            #plt.xlabel('Iteration')
            #plt.ylabel('Objective Value')
            #plt.title('Improvement of TSP Solution Over Iterations')
            #plt.show()
            if best_solution_value < ultra_solution_value:
                ultra_solution = best_solution
                ultra_solution_value = best_solution_value
                #print(f'New ultra solution found with value {ultra_solution_value} at seeding {seeding}')
        return ultra_solution

    def solve(self) -> list:
        # Trivial solution: visit nodes in the order they appear
        solution = self.optimal_solution()
        return solution