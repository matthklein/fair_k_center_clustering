import numpy as np
import scipy.sparse.csgraph
import matplotlib.pyplot as plt
import sys
import time
from algorithms import *
from algorithm_matroid_center import *


###############################################################
## Experiment shown in the left plot of Figure 3
###############################################################
def exp_compare_to_optimal_solution(nr_of_runs):

    print '-------------------------------------------'
    print 'exp_compare_to_optimal_solution'
    print '-------------------------------------------'

    setting_list=[[25, 2, 2, [2,2]],
                  [25, 2, 2, [4,2]],
                  [25, 3, 2, [2,2,2]],
                  [25, 3, 1, [5, 1, 1]],
                  [25, 4, 0, [2, 2, 2,2]],
                  [25, 4, 0, [3, 3, 1, 1]],
                  [25, 5, 0, [2, 2, 2, 1,1]]]


    pl_data=np.zeros((nr_of_runs,len(setting_list)))
    pl_data_MATROID = np.zeros((nr_of_runs, len(setting_list)))
    pl_time = np.zeros((nr_of_runs, len(setting_list)))
    pl_time_MATROID = np.zeros((nr_of_runs, len(setting_list)))
    settings_as_vec=np.array([])


    for tr,sl in enumerate(setting_list):

        n=sl[0]
        m=sl[1]
        nr_initially_given=sl[2]
        req_nr_per_sex=np.array(sl[3])

        print ''
        print 'n='+str(n)+', m='+str(m)+', (k_{S_1},...,k_{S_m})='+str(tuple(req_nr_per_sex))+', |C_0|='+str(nr_initially_given)

        settings_as_vec=np.hstack((settings_as_vec,np.array([n,m,nr_initially_given]),req_nr_per_sex))


        for rrr in np.arange(nr_of_runs):

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


            start = time.time()
            centers_approx = fair_k_center_APPROX(dmat, sexes, req_nr_per_sex, initially_given)
            cost_approx = np.amax(np.amin(dmat[np.ix_(np.hstack((centers_approx,initially_given)), np.arange(n))], axis=0))
            end = time.time()
            pl_time[rrr,tr] = end - start

            start = time.time()
            centers_approx_MATROID = MatCenter_binary_search_WithGivenCenters(dmat, sexes, req_nr_per_sex, initially_given)
            cost_approx_MATROID = np.amax(np.amin(dmat[np.ix_(np.hstack((centers_approx_MATROID, initially_given)), np.arange(n))], axis=0))
            end = time.time()
            pl_time_MATROID[rrr, tr] = end - start

            centers_exact, cl_exact, cost_exact = fair_k_center_exact(dmat, sexes, req_nr_per_sex,initially_given)


            fac = cost_approx / cost_exact
            pl_data[rrr,tr]=fac
            fac_MATROID = cost_approx_MATROID / cost_exact
            pl_data_MATROID[rrr, tr] = fac_MATROID
            print 'Approximation factor Alg. 4='+str(fac)+' ---- Approximation factor M.C.='+str(fac_MATROID)



    data=[]
    XT=[]
    for tr in np.arange(len(setting_list)):
        data.append(pl_data[:,tr])
        data.append(pl_data_MATROID[:, tr])
        XT.append('Alg. 4')
        XT.append('M.C.')
    fig, ax = plt.subplots(figsize=(12,4.5))
    ax.set_title('Simulated data with computable optimal solution, |S|='+str(n),fontsize=16)
    ax.boxplot(data)
    fig.tight_layout()
    plt.xticks([ggg for ggg in (1 + np.arange(2*len(setting_list)))],XT,fontsize=12)
    plt.ylabel('Approximation factor',fontsize=14)
    ylim = ax.get_ylim()
    new_ylim = (0.95,ylim[1])
    ax.set_ylim(new_ylim)
    fig.savefig('exp_compare_to_optimal_solution_APPROXFACTOR.pdf',bbox_inches='tight')
    plt.close()


    data = []
    XT = []
    for tr in np.arange(len(setting_list)):
        data.append(pl_time[:, tr])
        data.append(pl_time_MATROID[:, tr])
        XT.append('Alg. 4')
        XT.append('M.C.')
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_title('Simulated data with computable optimal solution, |S|=' + str(n), fontsize=16)
    ax.boxplot(data)
    fig.tight_layout()
    plt.xticks([ggg for ggg in (1 + np.arange(2 * len(setting_list)))], XT, fontsize=12)
    plt.ylabel('Running time [s]', fontsize=14)
    fig.savefig('exp_compare_to_optimal_solution_TIME.pdf', bbox_inches='tight')
    plt.close()







###############################################################
## Experiment shown in the right plot of Figure 3 and in
## Figure 9 in Appendix B
###############################################################
def exp_compare_to_each_other(nr_of_runs):

    print '-------------------------------------------'
    print 'exp_compare_to_each_other'
    print '-------------------------------------------'

    setting_list=[[50, 5, 0, [4,4,4,4,4]],
                  [100, 5, 0, [4,4,4,4,4]],
                  [150, 5, 0, [4, 4, 4, 4, 4]],
                  [200, 5, 0, [4, 4, 4, 4, 4]],
                  [250, 5, 0, [4, 4, 4, 4, 4]]]


    n_list=np.array([])

    pl_data=np.zeros((nr_of_runs,len(setting_list)))
    pl_data_MATROID = np.zeros((nr_of_runs, len(setting_list)))
    pl_time = np.zeros((nr_of_runs, len(setting_list)))
    pl_time_MATROID = np.zeros((nr_of_runs, len(setting_list)))
    settings_as_vec=np.array([])


    for tr,sl in enumerate(setting_list):

        n=sl[0]
        n_list=np.hstack((n_list,n))
        m=sl[1]
        nr_initially_given=sl[2]
        req_nr_per_sex=np.array(sl[3])

        print ''
        print 'n='+str(n)

        settings_as_vec=np.hstack((settings_as_vec,np.array([n,m,nr_initially_given]),req_nr_per_sex))


        for rrr in np.arange(nr_of_runs):

            indi_sexes = 0
            while indi_sexes==0:
                sexes=np.random.randint(m,size=n)

                elem_per_sex = np.zeros(m, dtype=int)
                for ell in np.arange(m):
                    elem_per_sex[ell] = np.sum(sexes == ell)

                if np.sum(elem_per_sex >= req_nr_per_sex) == m:
                    indi_sexes = 1

            initially_given = np.random.choice(n, size=nr_initially_given,replace=False)

            indi_dmat=0
            while indi_dmat==0:
                dmat=np.random.binomial(1,2*np.log(n)/n,(n,n))*np.random.randint(1,high=100+1,size=(n,n))+0.0
                dmat=np.triu(dmat,1)
                dmat=dmat+dmat.T
                scipy.sparse.csgraph.floyd_warshall(dmat,directed=False,overwrite=True)
                if not np.any(np.isinf(dmat)):
                    indi_dmat=1


            start = time.time()
            centers_approx = fair_k_center_APPROX(dmat, sexes, req_nr_per_sex, initially_given)
            cost_approx = np.amax(np.amin(dmat[np.ix_(np.hstack((centers_approx,initially_given)), np.arange(n))], axis=0))
            end = time.time()
            pl_data[rrr, tr] = cost_approx
            pl_time[rrr,tr] = end - start

            start = time.time()
            centers_approx_MATROID = MatCenter_binary_search_WithGivenCenters(dmat, sexes, req_nr_per_sex, initially_given)
            cost_approx_MATROID = np.amax(np.amin(dmat[np.ix_(np.hstack((centers_approx_MATROID, initially_given)), np.arange(n))], axis=0))
            end = time.time()
            pl_data_MATROID[rrr, tr] = cost_approx_MATROID
            pl_time_MATROID[rrr, tr] = end - start

            print 'Cost Alg. 4='+str(cost_approx)+', Time Alg. 4='+str(pl_time[rrr,tr])+' ---- Cost M.C.='+str(cost_approx_MATROID)+\
                ', Time M.C.='+str(pl_time_MATROID[rrr, tr])



    data=[]
    XT=[]
    for tr in np.arange(len(setting_list)):
        data.append(pl_data[:,tr])
        data.append(pl_data_MATROID[:, tr])
        XT.append('Alg. 4')
        XT.append('M.C.')
    fig, ax = plt.subplots(figsize=(12,4.5))
    ax.set_title('Simulated data, m=' + str(m) + ', k=' + str(np.sum(req_nr_per_sex)), fontsize=16)
    ax.boxplot(data)
    fig.tight_layout()
    plt.xticks([ggg for ggg in (1 + np.arange(2*len(setting_list)))],XT,fontsize=12)
    plt.ylabel('Cost',fontsize=14)
    ylim = ax.get_ylim()
    new_ylim = (0.95,ylim[1])
    ax.set_ylim(new_ylim)
    fig.savefig('exp_compare_to_each_other_COST.pdf',bbox_inches='tight')
    plt.close()



    plt.figure(figsize=(7.5, 4.5))
    plt.plot(n_list, np.mean(pl_time, 0), label='Alg. 4', marker="x", color='b')
    plt.plot(n_list, np.mean(pl_time_MATROID, 0), label='M.C.', marker="x", color='r')
    plt.plot(n_list,
             (np.mean(pl_time_MATROID, 0)[0] / (((n_list[0]) ** 2) * np.log(n_list[0]))) * (n_list ** 2) * np.log(
                 n_list), label=r'~ $|S|^2 \cdot \ln(|S|)$', linestyle="--", color='m')
    plt.plot(n_list, (np.mean(pl_time_MATROID, 0)[0] / ((n_list[0]) ** 2.5)) * (n_list ** 2.5),
             label=r'~ $|S|^{5/2}$', linestyle="--", color='g')

    plt.title('Simulated data, m=' + str(m) + ', k=' + str(np.sum(req_nr_per_sex)), fontsize=16)
    plt.legend()
    plt.xlabel('|S|', fontsize=14)
    plt.ylabel('Running time [s]', fontsize=14)
    plt.savefig('exp_compare_to_each_other_TIME.pdf', bbox_inches='tight')
    plt.close()






if __name__ == "__main__":
    if len(sys.argv)>1:
        number_of_runs=int(sys.argv[1])
    else:
        number_of_runs=10

    exp_compare_to_optimal_solution(number_of_runs)
    exp_compare_to_each_other(number_of_runs)
