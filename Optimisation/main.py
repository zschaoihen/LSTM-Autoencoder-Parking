# -*- coding: utf-8 -*-

from OfficerMain import *
import time

interval_time = 1  # time for police to deal with violations, type(int)
speed = 70  # walk speed, type(int)
startTime = 420  # start time, type(int)
endTime = 1140  # end time, type(int)

distanceFileName = 'data/EXData/dis_all.csv'
vioFileName1 = 'data/EXData/oneyear_month/JanToOct/testdata/N20161121.csv'
vioFileName2 = 'data/EXData/oneyear_month/JanToOct/testdata/N20161122.csv'
vioFileName3 = 'data/EXData/oneyear_month/JanToOct/testdata/N20161123.csv'
vioFileName4 = 'data/EXData/oneyear_month/JanToOct/testdata/N20161124.csv'
vioFileName5 = 'data/EXData/oneyear_month/JanToOct/testdata/N20161125.csv'
vioFileName6 = 'data/EXData/oneyear_month/JanToOct/testdata/N20161126.csv'
vioFileName7 = 'data/EXData/oneyear_month/JanToOct/testdata/N20161127.csv'
costtimeFileName = 'data/EXData/cost_all.csv'

Prob_FileName = 'data/EXData/oneyear_month/JanToOct/Clu_AK/clu_Prob_K_6.csv'
Clus_FileName = 'data/EXData/oneyear_month/JanToOct/Clu_AK/cluster_K_6.csv'

distance_data = pd.read_csv(distanceFileName, index_col=0)  # distance of two points
violation_data1 = pd.read_csv(vioFileName1, index_col=0)  # violation records
violation_data2 = pd.read_csv(vioFileName2, index_col=0)  # violation records
violation_data3 = pd.read_csv(vioFileName3, index_col=0)  # violation records
violation_data4 = pd.read_csv(vioFileName4, index_col=0)  # violation records
violation_data5 = pd.read_csv(vioFileName5, index_col=0)  # violation records
violation_data6 = pd.read_csv(vioFileName6, index_col=0)  # violation records
violation_data7 = pd.read_csv(vioFileName7, index_col=0)  # violation records
listviolation = []
listviolation.append(violation_data1)
listviolation.append(violation_data2)
listviolation.append(violation_data3)
listviolation.append(violation_data4)
listviolation.append(violation_data5)
listviolation.append(violation_data6)
listviolation.append(violation_data7)
cost_time_data = pd.read_csv(costtimeFileName, index_col=0)  # cost time of two points
prob_data = pd.read_csv(Prob_FileName, index_col=0)
cluster_data = pd.read_csv(Clus_FileName, index_col=0)
start_point = '13028N'  # start point
result_table = []
result_table_oneday = []
dis_table = []
dis_table_oneday = []

def main():
    """Use four methods to find the paths separately """
    for j in range(1, 2):
        result_table_oneday.clear()
        dis_table_oneday.clear()
        if j > 5:
            week = '2'
        else:
            week = '1'  # 1 is weekday and 2 is weekend
        violation_data =listviolation[j-1]
        print('date:', j)
        officer = Officer(distance_data, violation_data, speed, cost_time_data, start_point, startTime, endTime,
                          interval_time, prob_data, cluster_data, week, j)
        for algorithm in range(4, 5):  # '1':greedy '2':sa '3':ga '4':aco
            time_start = time.time()
            print(algorithm)
            result, distance = officer.find_path(algorithm)
            result_table_oneday.append(copy.deepcopy(result[0]))
            result_table_oneday.append(copy.deepcopy(result[1]))
            dis_table_oneday.append(copy.deepcopy(distance[0]))
            time_end = time.time()
            print(result)
            print('totally cost:', time_end - time_start)
        result_table.append(copy.deepcopy(result_table_oneday))
        dis_table.append(copy.deepcopy(dis_table_oneday))
    test2 = pd.DataFrame(data=result_table)
    test2.to_csv('actual_result.csv')
    test3 = pd.DataFrame(data=dis_table)
    test3.to_csv('actual_distance.csv')

if __name__ == '__main__':
    main()

