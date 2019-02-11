import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import sys
from algorithms import *
import os.path
import requests
import pandas

#####################################################################################
### Experiments using the Adult data set from the UCI Machine Learning Repository ###
### https://archive.ics.uci.edu/ml/datasets/adult                                 ###
#####################################################################################

###############################################################
## Experiment shown in middle and right plot of Figure 6
###############################################################
def exp_comparison_heuristics_adult_data_set(nr_of_runs,race_is_sensitive_attribute):
# if race_is_sensitive_attribute==1, then we use race as sensitive attribute, otherwise we use gender

    if race_is_sensitive_attribute==1:
        m = 5
    else:
        m=2

    print '-------------------------------------------'
    print 'exp_comparison_heuristics_adult_data_set with '+str(m)+' groups'
    print '-------------------------------------------'


    n = 25000
    nr_initially_given = 100

    if (not os.path.exists('adult.data')):
        print('Adult data set does not exist in current folder --- Have to download it')
        r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', allow_redirects=True)
        if r.status_code == requests.codes.ok:
            print('Download successful')
        else:
            print('Could not download Adult data set - please download it manually')
            sys.exit()
        open('adult.data', 'wb').write(r.content)


    df=pandas.read_csv('adult.data', sep=',',header=None)
    df=df[:n]

    if race_is_sensitive_attribute==1:
        sens_attr = 8
        sex = df[sens_attr]
        df = df.drop(columns=[sens_attr])
        sens_attributes = list(set(sex.astype(str).values))   # =[' Asian-Pac-Islander', ' White', ' Other', ' Amer-Indian-Eskimo', ' Black']
        sex_num = np.zeros(n, dtype=int)
        for rrr,ttt in enumerate(sens_attributes):
            sex_num[sex.astype(str).values == ttt]=rrr

        m=len(sens_attributes)   #m=5
        req_nr_per_sex=np.repeat(50, m)

    else:
        sens_attr = 9
        sex = df[sens_attr]
        sens_attributes = list(set(sex.astype(str).values))   # =[' Male', ' Female']
        df = df.drop(columns=[sens_attr])
        sex_num = np.zeros(n, dtype=int)
        sex_num[sex.astype(str).values == sens_attributes[1]] = 1

        m = len(sens_attributes)  # m=2
        req_nr_per_sex = np.repeat(200, m)


    #dropping non-numerical features and normalizing data
    cont_types=np.where(df.dtypes=='int')[0]   # =[0,2,4,9,10,11]
    df = df.iloc[:,cont_types]
    data = np.array(df.values, dtype=float)
    data = scale(data, axis=0)

    dmat = pairwise_distances(data,metric='l1')

    plot_data = np.zeros((nr_of_runs, 3))


    for rrr in np.arange(nr_of_runs):

        print 'run=', rrr

        initially_given = np.random.choice(n, size=nr_initially_given, replace=False)

        centers_approx = fair_k_center_APPROX(dmat, sex_num, req_nr_per_sex, initially_given)
        cost_approx = np.amax(
            np.amin(dmat[np.ix_(np.hstack((centers_approx, initially_given)), np.arange(n))], axis=0))

        centers_heuristic1 = heuristic_greedy_on_each_group(dmat, sex_num, req_nr_per_sex, initially_given)
        cost_heuristic1 = np.amax(
            np.amin(dmat[np.ix_(np.hstack((centers_heuristic1, initially_given)), np.arange(n))], axis=0))

        centers_heuristic2 = heuristic_greedy_till_constraint_is_satisfied(dmat, sex_num, req_nr_per_sex,
                                                                           initially_given)
        cost_heuristic2 = np.amax(
            np.amin(dmat[np.ix_(np.hstack((centers_heuristic2, initially_given)), np.arange(n))], axis=0))

        plot_data[rrr, 0] = cost_approx
        plot_data[rrr, 1] = cost_heuristic1
        plot_data[rrr, 2] = cost_heuristic2



    data = [plot_data[:, ccc] for ccc in np.arange(3)]
    fig, ax = plt.subplots(figsize=(3.4, 4.5))
    ax.set_title('Adult data set, |S|=' + str(n), fontsize=16)
    ax.boxplot(data)
    fig.tight_layout()
    plt.xticks([ggg for ggg in (1 + np.arange(3))], ['Our Alg.', 'Heur. A', 'Heur. B'], fontsize=12)
    plt.ylabel('Cost', fontsize=14)

    fig.savefig('plot_exp_comparison_heuristics_adult_data_set_m='+str(m)+'.pdf', bbox_inches='tight')
    plt.close()





###############################################################
## Experiment shown in middle and right plot of Figure 5
###############################################################
def exp_comparison_greedy_strategy_adult_data_set(nr_of_runs,race_is_sensitive_attribute):

    if race_is_sensitive_attribute==1:
        m = 5
    else:
        m=2

    print '-------------------------------------------'
    print 'exp_comparison_greedy_strategy_adult_data_set with '+str(m)+' groups'
    print '-------------------------------------------'


    n = 25000
    nr_initially_given = 100

    if (not os.path.exists('adult.data')):
        print('Adult data set does not exist in current folder --- Have to download it')
        r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', allow_redirects=True)
        if r.status_code == requests.codes.ok:
            print('Download successful')
        else:
            print('Could not download Adult data set - please download it manually')
            sys.exit()
        open('adult.data', 'wb').write(r.content)

    df = pandas.read_csv('adult.data', sep=',', header=None)
    df = df[:n]

    if race_is_sensitive_attribute == 1:
        sens_attr = 8
        sex = df[sens_attr]
        df = df.drop(columns=[sens_attr])
        sens_attributes = list(set(
            sex.astype(str).values))  # =[' Asian-Pac-Islander', ' White', ' Other', ' Amer-Indian-Eskimo', ' Black']
        sex_num = np.zeros(n, dtype=int)
        for rrr, ttt in enumerate(sens_attributes):
            sex_num[sex.astype(str).values == ttt] = rrr

        m = len(sens_attributes)  # m=5
        req_nr_per_sex = np.repeat(50, m)

    else:
        sens_attr = 9
        sex = df[sens_attr]
        sens_attributes = list(set(sex.astype(str).values))  # =[' Male', ' Female']
        df = df.drop(columns=[sens_attr])
        sex_num = np.zeros(n, dtype=int)
        sex_num[sex.astype(str).values == sens_attributes[1]] = 1

        m = len(sens_attributes)  # m=2
        req_nr_per_sex = np.repeat(200, m)

    # dropping non-numerical features and normalizing data
    cont_types = np.where(df.dtypes == 'int')[0]  # =[0,2,4,9,10,11]
    df = df.iloc[:, cont_types]
    data = np.array(df.values, dtype=float)
    data = scale(data, axis=0)

    dmat = pairwise_distances(data, metric='l1')

    plot_data = np.zeros((nr_of_runs, 3))


    for rrr in np.arange(nr_of_runs):

        print 'run=',rrr

        initially_given = np.random.choice(n, size=nr_initially_given, replace=False)

        centers_approx = fair_k_center_APPROX(dmat, sex_num, req_nr_per_sex, initially_given)
        cost_approx = np.amax(
            np.amin(dmat[np.ix_(np.hstack((centers_approx, initially_given)), np.arange(n))], axis=0))

        centers_greedy = k_center_greedy_with_given_centers(dmat, np.sum(req_nr_per_sex), initially_given)
        cost_greedy = np.amax(
            np.amin(dmat[np.ix_(np.hstack((centers_greedy, initially_given)), np.arange(n))], axis=0))

        plot_data[rrr, 0] = cost_approx
        plot_data[rrr, 1] = cost_greedy

        if m == 2:
            plot_data[rrr, 2] = np.abs(np.sum(sex_num[centers_greedy] == 0) - np.sum(sex_num[centers_greedy] == 1))
        else:
            maxdev = 0
            for der in np.arange(m):
                for das in np.arange(m):
                    maxdev = np.max([maxdev, np.abs(
                        np.sum(sex_num[centers_greedy] == der) - np.sum(sex_num[centers_greedy] == das))])
            plot_data[rrr, 2] = maxdev



    if m == 2:
        data = [plot_data[:, ccc] for ccc in np.arange(2)]

        fig = plt.figure()
        st = fig.suptitle('Adult data set, |S|=|S$_1$|+|S$_2$|=' + str(n), fontsize=16)

        ax1 = fig.add_subplot(121)
        ax1.boxplot(data)
        plt.xticks([1, 2], ['Our Alg.', 'Unfair Greedy'], fontsize=12)
        plt.ylabel('Cost', fontsize=14)

        ax2 = fig.add_subplot(122)
        ax2.boxplot(plot_data[:, 2])
        plt.xticks([1], ['Unfair Greedy'], fontsize=12)
        plt.ylabel('|# centers in S$_1$ - # centers in S$_2$|', fontsize=14)

        fig.tight_layout()

        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)

        fig.savefig('exp_comparison_greedy_strategy_adult_data_set_m='+str(m)+'.pdf', bbox_inches='tight')
        plt.close()

    else:
        data = [plot_data[:, ccc] for ccc in np.arange(2)]

        fig = plt.figure()
        st = fig.suptitle('Adult data set, |S|=|S$_1$|+...+|S$_5$|=' + str(n), fontsize=16)

        ax1 = fig.add_subplot(121)
        ax1.boxplot(data)
        plt.xticks([1, 2], ['Our Alg.', 'Unfair Greedy'], fontsize=12)
        plt.ylabel('Cost', fontsize=14)

        ax2 = fig.add_subplot(122)
        ax2.boxplot(plot_data[:, 2])
        plt.xticks([1], ['Unfair Greedy'], fontsize=12)
        plt.ylabel('max$_{i,j}$ |# centers in S$_i$ - # centers in S$_j$|', fontsize=13)

        fig.tight_layout()

        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)
        fig.savefig('exp_comparison_greedy_strategy_adult_data_set_m='+str(m)+'.pdf', bbox_inches='tight')
        plt.close()





if __name__ == "__main__":
    if len(sys.argv)>1:
        number_of_runs=int(sys.argv[1])
    else:
        number_of_runs=10

    exp_comparison_heuristics_adult_data_set(number_of_runs,0)
    exp_comparison_heuristics_adult_data_set(number_of_runs, 1)
    exp_comparison_greedy_strategy_adult_data_set(number_of_runs, 0)
    exp_comparison_greedy_strategy_adult_data_set(number_of_runs,1)
