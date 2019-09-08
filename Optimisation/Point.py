# -*- coding: utf-8 -*-

import math

class Point:
    """Point on map for each parking slot """
    def __init__(self, id_=-1, marker=None):
        self.id_ = id_
        self.marker = marker
        self.violated = False
        self.vioRecords = []

    def calculate_probability(self, current_time, arrive_time, day_state, prob_data, cluster_data):
        """Calculate leveling probability for a arriving point"""
        probability = 0.0
        for record in self.vioRecords:
            if current_time < record.vioTime or record.isCheck is True or current_time > record.depTime:
                continue
            else:
                index = arrive_time - record.vioTime
                """
                a = 1 means use the average probability
                a = 2 means use the probability after k-means or DBSCAN
                a = 3 means use the probability e^(-10/x)
                """
                a = 3
                if a == 1:
                    if index > 500:
                        probability = 0
                    else:
                        probability = prob_data.loc[self.marker, str(index - 1)]
                elif a == 2:
                    if record.vioTime < 721 and day_state == '1':
                        probability_state = cluster_data.loc[self.marker, 'weekday_mor']
                    elif record.vioTime > 720 and day_state == '1':
                        probability_state = cluster_data.loc[self.marker, 'weekday_aft']
                    elif record.vioTime < 721 and day_state == '2':
                        probability_state = cluster_data.loc[self.marker, 'weekend_mor']
                    else:
                        probability_state = cluster_data.loc[self.marker, 'weekend_aft']
                    if index > 500:
                        probability = 0
                    else:
                        probability = prob_data.loc[probability_state, str(index - 1)]
                else:
                    probability = math.exp(- index/10)
                return probability
        return probability
