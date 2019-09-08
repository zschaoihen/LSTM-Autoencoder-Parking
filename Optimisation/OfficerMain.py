# -*- coding: utf-8 -*-

import copy
import numpy as np
import math
import random
import time
from Point import Point
from AntMain import Ant
import pandas as pd


class ViolatedRecord:
    """Violated record history for one point"""
    def __init__(self, marker, vio_time, dep_time, is_check=False):
        """
        :param marker: the name of parking spot
        :param vioTime: the violation time of car in this parking spot
        :param depTime: the departure time of car in this parking spot
        :param isCheck: mark whether this records are be capture
        """
        self.marker = marker
        self.vioTime = vio_time
        self.depTime = dep_time
        self.isCheck = is_check


class Officer:
    """Find path """
    SA_STOPPiNG_ITER = 5000  # stop iteration of SA, type(int)
    SA_A = 0.995  # Annealing coefficient, type(float)
    SA_STOPPING_TEMP = 0.00001  # stop temperature of SA, type(float)

    CROSS_RATE = 0.7  # crossover probability of GA, type(float)
    MUTATE_RATE = 0.3  # mutation probability of GA, type(float)
    POP_SIZE = 500  # population size of GA, type(int)
    N_GENERATION = 50  # generation size of GA, type(int)

    ANT_COUNT = 10  # ant count of ACO, type(int)
    ACO_GENERATION = 20  # generation size of ACO, type(int)
    ALPHA = 1.0  # weight of pheromones, type(float)
    BETA = 10.0  # weight of visibility, type(float)
    RHO = 0.5  # pheromone evaporation coefficient, type(float)
    Q = 10  # pheromone enhancement coefficient, type(int)

    def __init__(self, distance_data, violation_list, walk_speed, cost_time_data, start_point,
                 start_time, end_time, interval_time, prob_data, cluster_data, day_state, index):
        self.walkingSpeed = walk_speed  # walk speed of police, type(int)
        self.startTime = start_time  # start time of find path, type(int)
        self.startPoint = copy.deepcopy(start_point)  # start point of find path
        self.endTime = end_time  # end time of find path, type(int)
        self.saveTime = 0  # save time of the final path, type(int)
        self.intervalTime = interval_time  # time for police to deal with violations, type(int)
        self.distanceData = distance_data   # distance of two points
        self.parkSpotData = distance_data.index.tolist()  # park plots
        self.costTime = cost_time_data  # cost time of two points
        self.point = self.get_point(violation_list, self.parkSpotData)  # park plots with its violation records
        self.unChangePoint = copy.deepcopy(self.point)  # unchanging park plots with its violation records
        self.m_path = []  # final path
        self.arrive = []  # arrive time of each point of the best solution
        self.totalProbability = []  # capture probability of each point of the best solution
        self.totalDistance = []  # total distance of the best solution
        self.benefit = 0  # benefits of the best solution

        self.adjust_arrive = []  # arrive time of the adjust solution
        self.adjust_candidate = []  # access point of the adjust solution
        self.adjust_probability = []  # capture probability of each point of the adjust solution
        self.provisionally_time = []  # somewhere need save provisionally arrive time of provisionally solution
        self.provisionally_prob = []  # somewhere need save provisionally capture probability of provisionally solution
        self.provisionally_path = []  # somewhere need save provisionally solution
        self.solution = []  # mutation results of GA or candidate next points of ACO

        self.sa_T = 0  # initial temperature of SA
        self.cur_path = []  # current solution of SA or GA
        self.cur_arrive = []  # current arrive time of each points of current solution of GA or SA
        self.ga_cur_benefit = []  # current benefits of population size solution of GA
        self.cur_benefits = 0  # current benefits of current solution of SA or ACO
        self.cur_save_time = 0  # current save time of current solution of SA
        self.cur_probability = []  # current capture probability of current solution of SA
        self.iteration = 0  # current iteration of SA

        self.pheromone = copy.deepcopy(self.costTime)  # initialize the pheromone table of ACO

        self.prob_data = prob_data
        self.cluster_data = cluster_data
        self.day_state = day_state
        self.index = str(index)

    def get_point(self, violation_list, park_spot_data):
        """Initial points """
        m_point = []  # park plots with its violation records
        vio = []  # violation time
        dep = []  # departure time
        # for index in range(len(park_spot_data)):
        for index in range(len(park_spot_data)):
            # print(index)
            point = Point(index, park_spot_data[index])
            for index1, row1 in violation_list.iterrows():
                if park_spot_data[index] == row1['street_marker']:
                    violation_time = self.time_transfer(row1['vio_time'])
                    departure_time = self.time_transfer(row1['departure_time'])
                    vio.append(violation_time)
                    dep.append(departure_time)
                    point.violated = True
            vio.sort()
            dep.sort()
            for i in range(len(vio)):
                point.vioRecords.append(ViolatedRecord(park_spot_data[index], vio[i], dep[i]))
            m_point.append(point)
            vio.clear()
            dep.clear()
        return m_point

    def time_transfer(self, time):
        """get total minutes from a time in a day"""
        hour_str, minute_str = time.split(':')
        hour = int(hour_str)
        minute = int(minute_str)
        return hour * 60 + minute

    def output(self, a):
        """Output the results of each method """
        test = pd.DataFrame(data=self.m_path)
        test.to_csv(a + self.index + '_path.csv')
        self.totalDistance.clear()
        self.calculate_total_distance()
        test1 = pd.DataFrame(data=self.totalDistance)
        test1.to_csv(a + self.index + '_dis.csv')
        output_benefit = []
        output_benefit.append(self.benefit)
        test2 = pd.DataFrame(data=output_benefit)
        test2.to_csv(a + self.index + '_pro.csv')
        test3 = pd.DataFrame(data=self.arrive)
        test3.to_csv(a + self.index + '_time.csv')

    def test(self):
        result_lenth = len(copy.deepcopy(self.m_path))
        path = copy.deepcopy(self.m_path)
        time_arrive = copy.deepcopy(self.arrive)
        point = copy.deepcopy(self.unChangePoint)
        tol_prob = 0
        for j in range(result_lenth):
            node = path[j]
            time = time_arrive[j]
            for node1 in point:
                if node == node1.marker:
                    for record in node1.vioRecords:
                        if record.vioTime < time < record.depTime:
                            if record.isCheck != True:
                                tol_prob += 1
                                record.isCheck = True
                                break
                            else:
                                break
        return_list = []
        return_list.append(len(path))
        return_list.append(tol_prob)
        return return_list

    def find_path(self, algorithm):
        """Find path"""
        self.m_path.clear()
        self.arrive.clear()
        self.totalProbability.clear()
        self.benefit = 0

        if algorithm == 1:
            self.greedy_find_path(algorithm)
        elif algorithm == 2:
            self.sa_find_path()
        elif algorithm == 3:
            self.ga_find_path()
        elif algorithm == 4:
            self.aco_find_path()

        return self.test(), self.totalDistance


    def greedy_find_path(self, method):
        """Greedy """
        consumed_time = self.startTime  # consumed time of total path
        self.m_path.clear()
        self.arrive.clear()
        self.arrive.append(consumed_time)
        self.m_path.append(self.startPoint)
        current_point = copy.deepcopy(self.startPoint)  # current point

        while consumed_time < self.endTime:
            next_point, max_p = self.move_next_point(current_point, consumed_time)
            if next_point == 'Null':
                consumed_time += 1
                self.saveTime += 1
            else:
                self.m_path.append(copy.deepcopy(next_point))
                travel_time = int(round(copy.deepcopy(self.costTime.loc[current_point, next_point]) +
                                        self.intervalTime))
                consumed_time += travel_time
                self.arrive.append(consumed_time)
                self.totalProbability.append(copy.deepcopy(max_p))
                self.update_node(copy.deepcopy(next_point), copy.deepcopy(consumed_time))
                current_point = copy.deepcopy(next_point)
        self.benefit = self.calculate_benefit(self.totalProbability)
        if method == 1:
            self.output(str(method))

    def move_next_point(self, current_point, current_time):
        """Select the next point """
        max_p, min_key = self.calculate_leaving_probability(current_point, current_time)
        if max_p < 0.001:
            next_point = 'Null'
        else:
            next_point = min_key
        return next_point, max_p

    def calculate_leaving_probability(self, current_point, current_time):
        """Select the next point by leaving probability """
        max_p = 0
        min_key = ''
        aver_max_p = 0  # maximum average probability
        max_arrive_time = 0

        for node in self.point:
            if node.violated:
                travel_time = int(round(self.costTime.loc[current_point, node.marker] + self.intervalTime))
                arrive_time = current_time + travel_time
                if arrive_time < 1440:
                    probability = node.calculate_probability(current_time, arrive_time, self.day_state, self.prob_data,
                                                             self.cluster_data)
                else:
                    probability = 0.0
                aver_pro = probability / travel_time
                if aver_max_p < aver_pro:
                    max_p = probability
                    min_key = node.marker
                    aver_max_p = aver_pro
        return max_p, min_key

    def calculate_benefit(self, prob):
        """Calculate the total benefit by total probability """
        benefit = 0
        for i in range(len(prob)):
            np.random.seed(0)
            a = np.random.rand()
            probability = 0
            if a < prob[i]:
                probability = 1
            benefit += probability
        return benefit

    def update_node(self, point, time):
        """Update the point violation records """
        for node in self.point:
            if point == node.marker:
                for record in node.vioRecords:
                    if record.vioTime < time < record.depTime:
                        record.isCheck = True
                        break

    def calculate_total_distance(self):
        """Calculate the total distance """
        distance = 0
        for i in range(len(self.m_path) - 1):
            if self.m_path[i + 1] == self.m_path[i]:
                a = 0
            else:
                a = self.distanceData[self.m_path[i + 1]][self.m_path[i]]
            distance += a
        self.totalDistance.append(distance)

    def sa_find_path(self):
        """Sa """
        self.sa_T = math.sqrt(len(self.parkSpotData))
        self.greedy_find_path('2')
        self.cur_path = self.m_path
        self.cur_benefits = self.benefit
        self.cur_probability = self.totalProbability
        self.cur_arrive = self.arrive
        self.cur_save_time = self.saveTime
        while self.sa_T >= Officer.SA_STOPPING_TEMP and self.iteration < Officer.SA_STOPPiNG_ITER:
            self.point = copy.deepcopy(self.unChangePoint)
            candidate = copy.deepcopy(list(self.cur_path))
            l = random.randint(2, len(self.cur_path) - 1)
            i = random.randint(1, l)
            candidate[i:l] = list(reversed(candidate[i:l]))
            new_candidate, probability, new_arrive_time, new_save_time = self.adjust_solution(candidate)
            self.sa_calculate_selected(candidate, probability, new_arrive_time, new_save_time)
            self.sa_T *= Officer.SA_A
            self.iteration += 1
        self.output('2')

    def adjust_solution(self, candidate):
        """Adjust the resulting solution """
        consumed_time = self.startTime
        self.adjust_arrive.clear()
        self.adjust_arrive.append(self.startTime)
        self.adjust_candidate.clear()
        self.adjust_probability.clear()
        cur_save_time = 0
        for i in range(1, len(candidate)):
            travel_time = copy.deepcopy(int(round(self.costTime.loc[candidate[i], candidate[i - 1]]
                                                  + self.intervalTime)))
            current_time = copy.deepcopy(consumed_time)
            consumed_time += travel_time
            cur_node = self.point[0]
            if consumed_time < self.endTime:
                for node in self.point:
                    if node.marker == candidate[i]:
                        cur_node = node
                        break
                cur_prob = cur_node.calculate_probability(current_time, consumed_time, self.day_state, self.prob_data,
                                                          self.cluster_data)
                cur_iter = 0
                while cur_prob < 0.001 and cur_iter < 5:
                    cur_iter += 1
                    current_time += 1
                    consumed_time += 1
                    cur_save_time += 1
                    cur_prob = cur_node.calculate_probability(current_time, consumed_time, self.day_state,
                                                              self.prob_data, self.cluster_data)
                self.adjust_probability.append(cur_prob)
                self.adjust_arrive.append(copy.deepcopy(consumed_time))
                if cur_prob != 0:
                    self.update_node(copy.deepcopy(candidate[i]), copy.deepcopy(consumed_time))
            else:
                consumed_time -= travel_time
        if len(candidate) == len(self.adjust_probability) + 1:
            self.adjust_candidate, new_probabilities, new_arrive_time, new_save_time = \
                self.adjust_solution_add(candidate, consumed_time, self.adjust_probability,
                                         self.adjust_arrive, cur_save_time)
        else:
            self.adjust_candidate = candidate[0: len(self.adjust_probability) + 1]
            new_probabilities = copy.deepcopy(self.adjust_probability)
            new_arrive_time = copy.deepcopy(self.adjust_arrive)
            new_save_time = cur_save_time
        new_candidate = copy.deepcopy(self.adjust_candidate)
        return new_candidate, new_probabilities, new_arrive_time, new_save_time

    def adjust_solution_add(self, candidate, consumed_time, probability, cur_arrive, cur_save_time):
        """Adjust the new solution if it can add access point"""
        current_point = candidate[-1]
        while consumed_time < self.endTime:
            next_point, max_p = self.move_next_point(current_point, consumed_time)
            if next_point == 'Null':
                consumed_time += 1
                cur_save_time += 1
            else:
                candidate.append(copy.deepcopy(next_point))
                travel_time = copy.deepcopy(int(round(self.costTime.loc[current_point, next_point] +
                                                      self.intervalTime)))
                consumed_time = consumed_time + travel_time
                probability.append(copy.deepcopy(max_p))
                cur_arrive.append(consumed_time)
                self.update_node(copy.deepcopy(next_point), copy.deepcopy(consumed_time))
                current_point = copy.deepcopy(next_point)
        return candidate, probability, cur_arrive, cur_save_time

    def sa_calculate_selected(self, candidate, probability, new_arrive_time, new_save_time):
        """Sa to select the best solution """
        candidate_fitness = self.calculate_benefit(probability)
        print("current fitness: ", self.cur_benefits)
        print('candidate fitness: ', candidate_fitness)
        print('best benefit: ', self.benefit)
        if candidate_fitness > self.cur_benefits:
            self.cur_benefits = candidate_fitness
            self.cur_path = candidate
            self.cur_probability = probability
            self.cur_arrive = new_arrive_time
            self.cur_save_time = new_save_time
            if candidate_fitness > self.benefit:
                self.benefit = candidate_fitness
                self.m_path = candidate
                self.totalProbability = probability
                self.arrive = new_arrive_time
                self.saveTime = new_save_time
        else:
            # Calculate the probability of receiving a solution worse than the current one
            ac = math.exp(-abs(candidate_fitness - self.cur_benefits) / self.sa_T)
            r = random.random()
            if r < ac:
                self.cur_benefits = candidate_fitness
                self.cur_path = candidate
                self.cur_probability = probability
                self.cur_arrive = new_arrive_time
                self.cur_save_time = new_save_time

    def ga_find_path(self):
        """GA """
        self.ga_init()
        for generation in range(Officer.N_GENERATION):
            self.point = copy.deepcopy(self.unChangePoint)
            self.ga_evolve()
            self.ga_find_best_solution()
        self.output('3')

    def ga_init(self):
        """Initialize the POP_SIZE group solution """
        self.ga_cur_benefit.clear()
        self.cur_path.clear()
        self.cur_arrive.clear()
        for i in range(Officer.POP_SIZE):
            self.provisionally_time.clear()
            self.provisionally_prob.clear()
            self.provisionally_path.clear()
            self.point = copy.deepcopy(self.unChangePoint)
            consumed_time = self.startTime
            self.provisionally_time.append(consumed_time)
            current_point = self.startPoint
            self.provisionally_path.append(self.startPoint)
            a = 0
            while consumed_time < self.endTime:
                rand_num = random.randint(0, len(self.parkSpotData) - 1)
                next_point = self.point[rand_num].marker
                travel_time = int(round(self.costTime.loc[current_point, next_point] + self.intervalTime))
                arrive_time = consumed_time + travel_time
                if arrive_time < self.endTime:
                    prob = self.point[rand_num].calculate_probability(consumed_time, arrive_time, self.day_state,
                                                                      self.prob_data, self.cluster_data)
                    if prob > 0 or a > 1122:
                        self.provisionally_prob.append(prob)
                        self.provisionally_path.append(next_point)
                        self.provisionally_time.append(arrive_time)
                        consumed_time = arrive_time
                        current_point = next_point
                        self.update_node(copy.deepcopy(next_point), copy.deepcopy(arrive_time))
                    else:
                        a += 1

                else:
                    consumed_time = self.endTime
            tol_pro = self.calculate_benefit(copy.deepcopy(self.provisionally_prob))
            self.ga_cur_benefit.append(copy.deepcopy(tol_pro))
            self.cur_path.append(copy.deepcopy(self.provisionally_path))
            self.cur_arrive.append(copy.deepcopy(self.provisionally_time))
            if i == 0:
                self.benefit = copy.deepcopy(copy.deepcopy(tol_pro))
                self.m_path = copy.deepcopy(copy.deepcopy(self.provisionally_path))
                self.arrive = copy.deepcopy(copy.deepcopy(self.provisionally_time))
            if self.benefit < tol_pro:
                self.benefit = copy.deepcopy(copy.deepcopy(tol_pro))
                self.m_path = copy.deepcopy(copy.deepcopy(self.provisionally_path))
                self.arrive = copy.deepcopy(copy.deepcopy(self.provisionally_time))

    def ga_evolve(self):
        """GA evolve process """
        selected_solution, selected_benefit, selected_arrive_time = self.ga_select()
        selected_solution_copy = copy.deepcopy(selected_solution)
        selected_benefit_copy = copy.deepcopy(selected_benefit)
        selected_arrive_time = copy.deepcopy(selected_arrive_time)
        self.provisionally_time.clear()
        self.provisionally_prob.clear()
        self.provisionally_path.clear()
        for j in range(len(selected_solution_copy)):  # for every parent
            parent = selected_solution_copy[j]
            child1, benefit, arrive_time = self.ga_crossover(parent, selected_solution_copy, selected_benefit_copy[j],
                                                             selected_arrive_time[j])
            child2 = child1.copy()
            child, cur_benefit, cur_arrive_time = self.ga_mutate(child2, benefit, arrive_time)
            self.provisionally_path.append(copy.deepcopy(child))
            self.provisionally_prob.append(copy.deepcopy(cur_benefit))
            self.provisionally_time.append(copy.deepcopy(cur_arrive_time))
        self.cur_path = copy.deepcopy(self.provisionally_path)
        self.ga_cur_benefit = copy.deepcopy(self.provisionally_prob)
        self.cur_arrive = copy.deepcopy(self.provisionally_time)

    def ga_select(self):
        """Screening results """
        fitness = np.array(self.ga_cur_benefit)
        self.provisionally_time.clear()
        self.provisionally_prob.clear()
        self.provisionally_path.clear()
        idx = np.random.choice(np.arange(Officer.POP_SIZE), size=Officer.POP_SIZE, replace=True,
                               p=fitness / fitness.sum())
        for i in range(len(idx)):
            self.provisionally_path.append(self.cur_path[idx[i]])
            self.provisionally_prob.append(self.ga_cur_benefit[idx[i]])
            self.provisionally_time.append(self.cur_arrive[idx[i]])
        return self.provisionally_path, self.provisionally_prob, self.provisionally_time

    def ga_crossover(self, parent, selected_solution_copy, benefit, arrive_time):
        """Crossover of GA"""
        if np.random.rand() < Officer.CROSS_RATE:
            parent = np.array(parent)
            i_ = np.random.randint(0, Officer.POP_SIZE, size=1)  # select another individual from pop
            cross_points = np.random.randint(0, 2, len(parent)).astype(np.bool)  # choose crossover points
            keep_plot = np.array(parent)[~cross_points]  # find the city number
            cross_points2 = np.random.randint(0, 2, len(selected_solution_copy[i_[0]])).astype(np.bool)
            swap_city = np.array(selected_solution_copy[i_[0]])[~cross_points2]
            parent = np.concatenate((keep_plot, swap_city)).tolist()
            if parent[0] != self.startPoint:
                parent.insert(0, self.startPoint)
            self.point = copy.deepcopy(self.unChangePoint)
            solution, probability, new_arrive_time, new_save_time = self.adjust_solution(parent)
            benefit = self.calculate_benefit(probability)
            arrive_time = new_arrive_time
        else:
            solution = parent
        return solution, benefit, arrive_time

    def ga_mutate(self, child, benefit, arrive_time):
        """Mutation of GA """
        self.solution.clear()
        for i in range(1, len(child)):
            if np.random.rand() < self.MUTATE_RATE:
                swap_point = np.random.randint(1, len(child) - 1)
                swap_a, swap_b = child[i], child[swap_point]
                child[i], child[swap_point] = swap_b, swap_a
                self.point = copy.deepcopy(self.unChangePoint)
                self.solution, probability, new_arrive_time, new_saveTime = self.adjust_solution(copy.deepcopy(child))
                benefit = self.calculate_benefit(probability)
                arrive_time = new_arrive_time
        if not self.solution:
            self.solution = child
        return self.solution, benefit, arrive_time

    def ga_find_best_solution(self):
        """Select the best solution of GA """
        # print(self.ga_cur_benefit)
        if self.benefit < self.ga_cur_benefit[self.ga_cur_benefit.index(max(self.ga_cur_benefit))]:
            self.benefit = self.ga_cur_benefit[self.ga_cur_benefit.index(max(self.ga_cur_benefit))]
            self.m_path = self.cur_path[self.ga_cur_benefit.index(max(self.ga_cur_benefit))]
            self.arrive = self.cur_arrive[self.ga_cur_benefit.index(max(self.ga_cur_benefit))]

    def pheromone_init(self):
        """Initialize the pheromone table """
        a = np.where(self.costTime == 0)
        b = np.where(self.costTime != 0)
        for i in range(len(a[1])):
            self.pheromone.values[a[0][i], a[1][i]] = 1
        for j in range(len(b[1])):
            self.pheromone.values[b[0][j], b[1][j]] = 1 / self.costTime.values[b[0][j], b[1][j]]

    def aco_find_path(self):
        """Find path by ACO"""
        self.pheromone_init()
        for gen in range(Officer.ACO_GENERATION):
            print("ACO gen: ", gen)
            # t0 = time.time()
            ants = [Ant(self) for i in range(Officer.ANT_COUNT)]  # Initialize ANT_COUNT ants
            # Each ant chooses the next plot by probability function and completes its own tour
            # t1 = time.time()
            # print("t1 - t0: ", (t1 - t0))
            for ant in ants:
                # t2 = time.time()
                self.point = copy.deepcopy(self.unChangePoint)
                self.provisionally_time.clear()
                self.provisionally_prob.clear()
                self.provisionally_path.clear()
                self.provisionally_path.append(self.startPoint)
                self.provisionally_time.append(self.startTime)
                point = self.point[random.randint(0, len(self.point) - 1)]  # Randomly select the second point to access
                current_time = self.startTime + copy.deepcopy(int(round(self.costTime.loc[self.startPoint, point.marker]
                                                                        + self.intervalTime)))
                current_probability = point.calculate_probability(self.startTime, current_time, self.day_state,
                                                                  self.prob_data, self.cluster_data)
                self.provisionally_prob.append(current_probability)
                self.provisionally_path.append(point.marker)
                self.provisionally_time.append(current_time)
                consumed_time = current_time
                # Update node violation record after access
                self.update_node(copy.deepcopy(point.marker), copy.deepcopy(consumed_time))
                while consumed_time < self.endTime:
                    self.solution.clear()
                    pro, consumed_time, self.provisionally_time = Ant.select_next(ant, consumed_time,
                                                                                  self.provisionally_path,
                                                                                  self.provisionally_prob,
                                                                                  self.provisionally_time,
                                                                                  copy.deepcopy(self.solution),
                                                                                  self.day_state, self.prob_data,
                                                                                  self.cluster_data)
                    if pro[-1] == -2:  # True if no nodes can arrive
                        self.provisionally_prob.pop()
                        break
                self.cur_benefits = self.calculate_benefit(copy.deepcopy(self.provisionally_prob))
                if self.cur_benefits > self.benefit:  # Select the best solution
                    self.benefit = self.cur_benefits
                    self.m_path = copy.deepcopy(self.provisionally_path)
                    self.arrive = copy.deepcopy(self.provisionally_time)

                # t3 = time.time()
                # Calculate the pheromone left behind by each ant
                Ant.update_pheromone_delta(ant, copy.deepcopy(self.provisionally_prob))
                # t4 = time.time()
            #     print("t3 - t2: ", (t1 - t0))
            #     print("t4 - t3: ", (t1 - t0))
            # t5 = time.time()
            # update pheromone
            self.update_pheromone(ants)
            # t6 = time.time()
            # print("t6 - t5: ", (t1 - t0))
        self.output('4')

    def update_pheromone(self, ants):
        """Update the pheromone of ants"""
        np.multiply(self.pheromone, self.RHO)  # Pheromones evaporation
        for ant in ants:  # Pheromone reinforcement
            np.add(self.pheromone, ant.pheromone_delta)
