import numpy as np
import scipy.sparse.csgraph
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import sys
from algorithms import *



###############################################################
## Experiment shown in Figure 4
###############################################################
def exp_approx_factor_artificial_data(nr_of_runs):

    print '-------------------------------------------'
    print 'exp_approx_factor_artificial_data'
    print '-------------------------------------------'

    setting_list = [ell for ell in 2 + np.arange(19)]
    plot_data = np.zeros((nr_of_runs, len(setting_list)))

    n = 10000
    initially_given = np.array([], dtype=int)

    centersTRUE = np.zeros((100, 2))
    ccc = 0
    for zzz in np.arange(10):
        for rrr in np.arange(10):
            centersTRUE[ccc, 0] = zzz
            centersTRUE[ccc, 1] = rrr
            ccc += 1


    for ccc, m in enumerate(setting_list):

        print 'm = ',m

        for rrr in np.arange(nr_of_runs):
            sexes = np.random.randint(m, size=n)
            points = np.random.normal(size=(n, 2))

            points_ce = np.random.choice(centersTRUE.shape[0], n)
            for zzz in np.arange(centersTRUE.shape[0]):
                cluster = np.where(points_ce == zzz)[0]
                radius_cluster = np.max(np.sum((points[cluster, :]) ** 2, axis=1) ** (0.5))
                points[cluster, :] = 0.5 * points[cluster, :] / radius_cluster + np.repeat(centersTRUE[zzz, :].reshape(1, 2),
                                                                 cluster.size, axis=0)

            sex_centersTRUE = np.random.randint(m,size=centersTRUE.shape[0])
            req_nr_per_sex = np.zeros(m,dtype=int)
            for ell in np.arange(m):
                req_nr_per_sex[ell] = np.sum(sex_centersTRUE == ell)

            points = np.vstack((points, centersTRUE))
            sexes = np.hstack((sexes, sex_centersTRUE))
            hh = np.random.permutation(sexes.size)
            sexes = sexes[hh]
            points = points[hh, :]
            dmat = pairwise_distances(points)

            centers_approx = fair_k_center_APPROX(dmat, sexes, req_nr_per_sex, initially_given)
            cost_approx = np.amax(
                np.amin(dmat[np.ix_(np.hstack((centers_approx, initially_given)), np.arange(dmat.shape[0]))], axis=0))
            cost_exact = 0.5
            fac = cost_approx / cost_exact
            plot_data[rrr, ccc] = fac

            print 'Approximation factor =',fac



    data = [plot_data[:, ccc] for ccc in np.arange(len(setting_list))]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_title('Simulated data with known optimal solution, S=' + str(dmat.shape[0]), fontsize=16)
    ax.boxplot(data)
    fig.tight_layout()
    plt.xticks([ggg for ggg in (1 + np.arange(len(setting_list)))],
               ['m=' + str(setting_list[ggg - 1]) for ggg in (1 + np.arange(len(setting_list)))], fontsize=12)
    plt.ylabel('Approximation factor', fontsize=14)

    fig.savefig('plot_exp_approx_factor_artificial_data.pdf', bbox_inches='tight')
    plt.close()





###############################################################
## Experiment shown in left plot of Figure 6
###############################################################
def exp_comparison_heuristics_artificial_data(nr_of_runs):

    print '-------------------------------------------'
    print 'exp_comparison_heuristics_artificial_data'
    print '-------------------------------------------'

    plot_data = np.zeros((nr_of_runs, 3))

    n = 2000
    m = 10
    nr_initially_given = 10
    req_nr_per_sex = np.repeat(4,m)


    for rrr in np.arange(nr_of_runs):

        print 'run=',rrr

        indi_sexes = 0
        while indi_sexes == 0:
            sexes = np.random.randint(m, size=n)

            elem_per_sex = np.zeros(m, dtype=int)
            for ell in np.arange(m):
                elem_per_sex[ell] = np.sum(sexes == ell)

            if np.sum(elem_per_sex >= req_nr_per_sex) == m:
                indi_sexes = 1

        initially_given = np.random.choice(n, size=nr_initially_given, replace=False)

        indi_dmat = 0
        while indi_dmat == 0:
            dmat = np.random.binomial(1, 2 * np.log(n) / n, (n, n)) * np.random.randint(1, high=100 + 1,
                                                                                        size=(n, n)) + 0.0
            dmat = np.triu(dmat, 1)
            dmat = dmat + dmat.T

            scipy.sparse.csgraph.floyd_warshall(dmat, directed=False, overwrite=True)

            if not np.any(np.isinf(dmat)):
                indi_dmat = 1


        centers_approx = fair_k_center_APPROX(dmat, sexes, req_nr_per_sex, initially_given)
        cost_approx = np.amax(
            np.amin(dmat[np.ix_(np.hstack((centers_approx, initially_given)), np.arange(n))], axis=0))

        centers_heuristic1 = heuristic_greedy_on_each_group(dmat, sexes, req_nr_per_sex, initially_given)
        cost_heuristic1 = np.amax(
            np.amin(dmat[np.ix_(np.hstack((centers_heuristic1, initially_given)), np.arange(n))], axis=0))

        centers_heuristic2 = heuristic_greedy_till_constraint_is_satisfied(dmat, sexes, req_nr_per_sex,
                                                                           initially_given)
        cost_heuristic2 = np.amax(
            np.amin(dmat[np.ix_(np.hstack((centers_heuristic2, initially_given)), np.arange(n))], axis=0))

        plot_data[rrr, 0] = cost_approx
        plot_data[rrr, 1] = cost_heuristic1
        plot_data[rrr, 2] = cost_heuristic2



    data = [plot_data[:, ccc] for ccc in np.arange(3)]
    fig, ax = plt.subplots(figsize=(3.4, 4.5))
    ax.set_title('Simulated data, |S|=' + str(n), fontsize=16)
    ax.boxplot(data)
    fig.tight_layout()
    plt.xticks([ggg for ggg in (1 + np.arange(3))], ['Our Alg.', 'Heur. A', 'Heur. B'], fontsize=12)
    plt.ylabel('Cost', fontsize=14)

    fig.savefig('plot_exp_comparison_heuristics_artificial_data.pdf', bbox_inches='tight')
    plt.close()





###############################################################
## Experiment shown in left plot of Figure 5
###############################################################
def exp_comparison_greedy_strategy_artificial_data(nr_of_runs):

    print '-------------------------------------------'
    print 'exp_comparison_greedy_strategy_artificial_data'
    print '-------------------------------------------'

    plot_data = np.zeros((nr_of_runs, 3))

    n = 2000
    m = 10
    nr_initially_given = 10
    req_nr_per_sex = np.repeat(4, m)



    for rrr in np.arange(nr_of_runs):

        print 'run=',rrr

        indi_sexes = 0
        while indi_sexes == 0:
            sexes = np.random.randint(m, size=n)

            elem_per_sex = np.zeros(m, dtype=int)
            for ell in np.arange(m):
                elem_per_sex[ell] = np.sum(sexes == ell)

            if np.sum(elem_per_sex >= req_nr_per_sex) == m:
                indi_sexes = 1

        initially_given = np.random.choice(n, size=nr_initially_given, replace=False)

        indi_dmat = 0
        while indi_dmat == 0:
            dmat = np.random.binomial(1, 2 * np.log(n) / n, (n, n)) * np.random.randint(1, high=100 + 1,
                                                                                        size=(n, n)) + 0.0
            dmat = np.triu(dmat, 1)
            dmat = dmat + dmat.T

            scipy.sparse.csgraph.floyd_warshall(dmat, directed=False, overwrite=True)

            if not np.any(np.isinf(dmat)):
                indi_dmat = 1


        centers_approx = fair_k_center_APPROX(dmat, sexes, req_nr_per_sex, initially_given)
        cost_approx = np.amax(
            np.amin(dmat[np.ix_(np.hstack((centers_approx, initially_given)), np.arange(n))], axis=0))

        centers_greedy = k_center_greedy_with_given_centers(dmat, np.sum(req_nr_per_sex), initially_given)
        cost_greedy = np.amax(
            np.amin(dmat[np.ix_(np.hstack((centers_greedy, initially_given)), np.arange(n))], axis=0))

        plot_data[rrr, 0] = cost_approx
        plot_data[rrr, 1] = cost_greedy

        if m == 2:
            plot_data[rrr, 2] = np.abs(np.sum(sexes[centers_greedy] == 0) - np.sum(sexes[centers_greedy] == 1))
        else:
            maxdev = 0
            for der in np.arange(m):
                for das in np.arange(m):
                    maxdev = np.max([maxdev, np.abs(
                        np.sum(sexes[centers_greedy] == der) - np.sum(sexes[centers_greedy] == das))])
            plot_data[rrr, 2] = maxdev



    if m == 2:
        data = [plot_data[:, ccc] for ccc in np.arange(2)]

        fig = plt.figure()
        st = fig.suptitle('Simulated data, |S|=|S$_1$|+|S$_2$|=' + str(n), fontsize=16)

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

        fig.savefig('exp_comparison_greedy_strategy_artificial_data.pdf', bbox_inches='tight')
        plt.close()

    else:
        data = [plot_data[:, ccc] for ccc in np.arange(2)]

        fig = plt.figure()
        st = fig.suptitle('Simulated data, |S|=|S$_1$|+...+|S$_{10}$|=' + str(n), fontsize=16)

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
        fig.savefig('exp_comparison_greedy_strategy_artificial_data.pdf', bbox_inches='tight')
        plt.close()





if __name__ == "__main__":
    if len(sys.argv)>1:
        number_of_runs=int(sys.argv[1])
    else:
        number_of_runs=10

    exp_approx_factor_artificial_data(number_of_runs)
    exp_comparison_heuristics_artificial_data(number_of_runs)
    exp_comparison_greedy_strategy_artificial_data(number_of_runs)
