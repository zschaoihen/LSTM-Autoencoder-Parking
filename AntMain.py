# -*- coding: utf-8 -*-

import random
import copy
import numpy as np


class Ant(object):
    def __init__(self, aco):
        """
        :param colony: some information of find path
        :param tabu: the path of this ant
        :param pheromone_delta: The pheromones left by the ant as it go this path
        """
        self.colony = aco
        self.tabu = []
        self.pheromone_delta = []

    def select_next(self, consumed_time: int, current_path: list, total_probability: list, current_arrive_time: list,
                    node: list, day_state, prob_data, cluster_data):
        """select the next node of each ant"""
        denominator = 0.0  # Sum of visibility and pheromone power of all optional points
        save_time = 0
        self.tabu = current_path
        current_point = self.tabu[-1]
        for i in range(len(self.colony.parkSpotData)):  # Select all available points
            next_point = self.colony.point[i].marker
            current_time = consumed_time + int(round(self.colony.costTime.loc[current_point, next_point] +
                                                     self.colony.intervalTime))
            if current_time < self.colony.endTime:  # True if this point can be selected
                current_prob = self.colony.point[i].calculate_probability(consumed_time, current_time, day_state,
                                                                          prob_data, cluster_data)
                unit_time_prob = current_prob / (self.colony.costTime.loc[current_point, next_point] +
                                                 self.colony.intervalTime)
                denominator += (self.colony.pheromone.loc[current_point, next_point] ** self.colony.ALPHA) * \
                               (unit_time_prob ** self.colony.BETA)
                c = [next_point, unit_time_prob, current_prob]
                node.append(copy.deepcopy(c))
        if denominator == 0:  # True if all available point's probability is 0
            consumed_time += 1
            save_time += 1
        elif len(node) != 0:  # True if capture possibility is not all 0
            # The probability of the current point reaching each possible next node
            candidate_point = [0 for i in range(len(node))]
            for i in range(len(node)):
                candidate_point[i] = self.colony.pheromone.loc[current_point, node[i][0]] ** self.colony.ALPHA * \
                                       node[i][1] ** self.colony.BETA / denominator
            # Select next node by probability roulette
            selected = node[0][0]
            pro = node[0][2]
            rand = random.random()
            for i, probability in enumerate(candidate_point):
                rand -= probability
                if rand <= 0:
                    selected = node[i][0]
                    pro = node[i][2]
                    break
            self.tabu.append(copy.deepcopy(selected))
            total_probability.append(copy.deepcopy(pro))
            consumed_time += int(round(self.colony.costTime.loc[current_point, selected] + self.colony.intervalTime))
            current_arrive_time.append(copy.deepcopy(consumed_time))
            # Update node violation record after access
            self.colony.update_node(copy.deepcopy(selected), copy.deepcopy(consumed_time))
        else:
            total_probability.append(-2)
        return total_probability, consumed_time, current_arrive_time

    def update_pheromone_delta(self, total_probability: list):
        """The pheromone left behind by each ant """
        self.pheromone_delta = copy.deepcopy(self.colony.costTime)
        np.multiply(self.pheromone_delta, 0)  # Initial each ant pheromone
        for i in range(1, len(self.tabu)):  # Calculate the pheromone left by ant
            first_node = self.tabu[i - 1]
            second_node = self.tabu[i]
            self.pheromone_delta.loc[first_node, second_node] += total_probability[i - 1] * self.colony.Q
