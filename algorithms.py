import numpy as np
import scipy.sparse.csgraph
import itertools



########################################################################################################################
def fair_k_center_exact(dmat,sexes,nr_centers_per_sex,given_centers):

#EXHAUSTIVE SEARCH TO EXACTLY SOLVE THE FAIR K-CENTER PROBLEM (2) --- ONLY WORKS FOR SMALL PROBLEM INSTANCES!!!
#dmat ... distance matrix of size nxn
#sexes ... integer-vector of length n with entries in 0,...,m-1, where m is the number of groups
#nr_centers_per_sex ... integer-vector of length m with entries in 0,...,k and sum over entries equaling k
#given centers ... integer-vector with entries in 0,...,n-1


    n = dmat.shape[0]
    m = nr_centers_per_sex.size
    k = np.sum(nr_centers_per_sex)

    cost = np.inf
    best_choice = []


    for mmm in itertools.combinations(np.arange(n),k):

        cluster_centers = np.array(mmm)

        curr_nr_clusters_per_sex = np.zeros(m)
        for ell in np.arange(m):
            curr_nr_clusters_per_sex[ell] = np.sum(sexes[cluster_centers]==ell)

        if sum(curr_nr_clusters_per_sex==nr_centers_per_sex)==m:
            curr_cost = np.amax(np.amin(dmat[np.ix_(np.hstack((cluster_centers,given_centers)), np.arange(n))],axis=0))
        else:
            curr_cost = np.inf


        if curr_cost<cost:
            cost = curr_cost
            best_choice = cluster_centers.copy()

    clustering = np.array([np.argmin(dmat[ell, np.hstack((best_choice,given_centers))]) for ell in np.arange(n)])

    return best_choice, clustering, cost
########################################################################################################################



########################################################################################################################
def k_center_greedy_with_given_centers(dmat,k,given_centers):

#IMPLEMENTATION OF ALGORITHM 1
#dmat ... distance matrix of size nxn
#k ... integer smaller than n
#given centers ... integer-vector with entries in 0,...,n-1


    n=dmat.shape[0]

    if k==0:
        cluster_centers = np.array([],dtype=int)
    else:
        if given_centers.size==0:
            cluster_centers = np.random.choice(n,1,replace=False)
            kk = 1
        else:
            cluster_centers = given_centers
            kk = 0

        distance_to_closest = np.amin(dmat[np.ix_(cluster_centers,np.arange(n))],axis=0)
        while kk<k:
            temp = np.argmax(distance_to_closest)
            cluster_centers = np.append(cluster_centers,temp)
            distance_to_closest = np.amin(np.vstack((distance_to_closest,dmat[temp,:])),axis=0)
            kk+=1

        cluster_centers = cluster_centers[given_centers.size:]


    return cluster_centers
########################################################################################################################



########################################################################################################################
def fair_k_center_APPROX(dmat,sexes,nr_centers_per_sex,given_centers):

#IMPLEMENTATION OF ALGORITHM 4
#dmat ... distance matrix of size nxn
#sexes ... integer-vector of length n with entries in 0,...,m-1, where m is the number of groups
#nr_centers_per_sex ... integer-vector of length m with entries in 0,...,k and sum over entries equaling k
#given centers ... integer-vector with entries in 0,...,n-1


    n = dmat.shape[0]
    m = nr_centers_per_sex.size
    k = np.sum(nr_centers_per_sex)


    if m==1:
        cluster_centers = k_center_greedy_with_given_centers(dmat, k, given_centers)

    else:
        cluster_centersTE = k_center_greedy_with_given_centers(dmat, k, given_centers)

        CURRENT_nr_clusters_per_sex = np.zeros(m, dtype=int)
        for ell in np.arange(k):
            CURRENT_nr_clusters_per_sex[sexes[cluster_centersTE[ell]]] += 1

        partition = np.array([np.argmin(dmat[ell, np.hstack((cluster_centersTE,given_centers))]) for ell in np.arange(n)])
        G,centersTE = swapping_graph(partition[partition<k], np.array([np.where(np.arange(n)[partition < k] ==
                            cluster_centersTE[ell])[0][0] for ell in np.arange(k)]),sexes[partition<k], nr_centers_per_sex)
        cluster_centersTE=np.arange(n)[partition < k][centersTE]


        if G.size==0:
            cluster_centers=cluster_centersTE
        else:
            new_data_set=np.array([],dtype=int)
            new_given_centersT=np.array([],dtype=int)
            for ell in np.arange(k):
                if np.isin(sexes[cluster_centersTE[ell]],G):
                    new_data_set=np.hstack((new_data_set,np.where(partition==ell)[0]))
                else:
                    new_given_centersT=np.hstack((new_given_centersT,cluster_centersTE[ell]))
            new_given_centers=np.hstack((new_given_centersT,given_centers))
            sexes_new = sexes[new_data_set]
            sexes_newT=np.zeros(new_data_set.size,dtype=int)
            cc=0
            for ell in G:
                sexes_newT[sexes_new==ell]=cc
                cc+=1
            new_data_set=np.hstack((new_data_set,new_given_centers))
            sexes_newT=np.hstack((sexes_newT,np.zeros(new_given_centers.size,dtype=int)))

            cluster_centers_rek=fair_k_center_APPROX(dmat[np.ix_(new_data_set,new_data_set)], sexes_newT,
                                    nr_centers_per_sex[G],np.arange(new_data_set.size-new_given_centers.size,new_data_set.size))


            new_given_centersT_additional= np.array([],dtype=int)
            for ell in np.setdiff1d(np.arange(m),G):
                if np.sum(sexes[new_given_centersT]==ell)<nr_centers_per_sex[ell]:
                    toadd=nr_centers_per_sex[ell]-np.sum(sexes[new_given_centersT]==ell)
                    toadd_pot=np.setdiff1d(np.where(sexes == ell)[0], new_given_centersT)
                    if toadd_pot.size>toadd:
                        new_given_centersT_additional=np.hstack((new_given_centersT_additional,toadd_pot[0:toadd]))
                    else:
                        new_given_centersT_additional = np.hstack((new_given_centersT_additional, toadd_pot))

            cluster_centers=np.hstack((new_given_centersT,new_given_centersT_additional,new_data_set[cluster_centers_rek]))


    return cluster_centers
########################################################################################################################



########################################################################################################################
def swapping_graph(partition,centers,sexes,nr_centers_per_sex):

#IMPLEMENTATION OF ALGORITHM 3
#partition ... integer-vector of length n with entries in 0 ... k-1
#centers ... integer-vector of length k with entries in 0 ... n-1
#sexes ... integer-vector of length n with entries in 0 ... m-1
#nr_centers_per_sex ... integer-vector of length m with entries in 0,...,k and sum over entries equaling k


    n = partition.size
    m = nr_centers_per_sex.size
    k = centers.size


    CURRENT_nr_clusters_per_sex = np.zeros(m, dtype=int)
    for ell in np.arange(k):
        CURRENT_nr_clusters_per_sex[sexes[centers[ell]]] += 1

    sex_of_assigned_center = sexes[partition]
    Adja = np.zeros((m, m))
    for ell in np.arange(n):
        Adja[sex_of_assigned_center[ell],sexes[ell]] = 1

    dmat,predec = scipy.sparse.csgraph.shortest_path(Adja, directed=True, return_predecessors=True)

    is_there_a_path=0
    for ell in np.arange(m):
        for zzz in np.arange(m):
            if ((CURRENT_nr_clusters_per_sex[ell]>nr_centers_per_sex[ell]) and (CURRENT_nr_clusters_per_sex[zzz]<nr_centers_per_sex[zzz])):
                if dmat[ell,zzz]!=np.inf:
                    path = np.array([zzz])
                    while path[0]!=ell:
                        path = np.hstack((predec[ell,path[0]],path))
                    is_there_a_path = 1
                    break
        if is_there_a_path==1:
            break



    while (is_there_a_path) and (np.sum(CURRENT_nr_clusters_per_sex==nr_centers_per_sex)<m):

        for hhh in np.arange(path.size - 1):
            for ell in np.arange(n):
                if (sexes[ell]==path[hhh+1]) and (sex_of_assigned_center[ell]==path[hhh]):
                    centers[partition[ell]] = ell
                    sex_of_assigned_center[partition==partition[ell]] = sexes[ell]
                    break
        CURRENT_nr_clusters_per_sex[path[0]] -= 1
        CURRENT_nr_clusters_per_sex[path[-1]] += 1


        Adja = np.zeros((m, m))
        for ell in np.arange(n):
            Adja[sex_of_assigned_center[ell], sexes[ell]] = 1

        dmat, predec = scipy.sparse.csgraph.shortest_path(Adja, directed=True, return_predecessors=True)

        is_there_a_path = 0
        for ell in np.arange(m):
            for zzz in np.arange(m):
                if ((CURRENT_nr_clusters_per_sex[ell] > nr_centers_per_sex[ell]) and (CURRENT_nr_clusters_per_sex[zzz] < nr_centers_per_sex[zzz])):
                    if dmat[ell, zzz] != np.inf:
                        path = np.array([zzz])
                        while path[0] != ell:
                            path = np.hstack((predec[ell, path[0]], path))
                        is_there_a_path = 1
                        break
            if is_there_a_path == 1:
                break




    if sum(CURRENT_nr_clusters_per_sex==nr_centers_per_sex)==m:
        return np.array([]),centers
    else:

        G = np.where(CURRENT_nr_clusters_per_sex > nr_centers_per_sex)[0]
        for ell in np.arange(m):
            for zzz in np.arange(m):
                if (((dmat[ell, zzz] != np.inf) and np.isin(ell, G)) and (not np.isin(zzz, G))):
                    G = np.hstack((G, zzz))


        return G,centers
########################################################################################################################



########################################################################################################################
def heuristic_greedy_on_each_group(dmat,sexes,nr_centers_per_sex,given_centers):

# IMPLEMENTATION OF HEURISTIC A AS DESCRIBED IN SECTION 5.3
# dmat ... distance matrix of size nxn
# sexes ... integer-vector of length n with entries in 0,...,m-1, where m is the number of groups
# nr_centers_per_sex ... integer-vector of length m with entries in 0,...,k and sum over entries equaling k
# given centers ... integer-vector with entries in 0,...,n-1

    n = dmat.shape[0]
    m = nr_centers_per_sex.size

    cluster_centers=np.array([],dtype=int)

    for ell in np.arange(m):
        subgroup=np.where(sexes==ell)[0]
        given_centers_subgroup=np.where(np.isin(subgroup,given_centers))[0]

        cent_subgroup=k_center_greedy_with_given_centers(dmat[np.ix_(subgroup, subgroup)],
                                                               nr_centers_per_sex[ell], given_centers_subgroup)

        cluster_centers=np.hstack((cluster_centers,subgroup[cent_subgroup]))

    return cluster_centers
########################################################################################################################



########################################################################################################################
def heuristic_greedy_till_constraint_is_satisfied(dmat,sexes,nr_centers_per_sex,given_centers):

# IMPLEMENTATION OF HEURISTIC B AS DESCRIBED IN SECTION 5.3
# dmat ... distance matrix of size nxn
# sexes ... integer-vector of length n with entries in 0,...,m-1, where m is the number of groups
# nr_centers_per_sex ... integer-vector of length m with entries in 0,...,k and sum over entries equaling k
# given centers ... integer-vector with entries in 0,...,n-1
#SINCE WE DO NOT REQUIRE THE GENERAL CASE IN OUR EXPERIMENTS AND IT ALLOWS FOR A SLIGHTLY SIMPLER CODE,
#HERE WE ASSUME THAT EITHER length(given_centers)>0 OR nr_centers_per_sex only[i]>0 FOR ALL i

    n = dmat.shape[0]
    m = nr_centers_per_sex.size
    k=np.sum(nr_centers_per_sex)

    current_nr_per_sex=np.zeros(m)

    if k==0:
        cluster_centers = np.array([], dtype=int)
    else:
        if given_centers.size==0:
            cluster_centers=np.random.choice(n,1,replace=False)
            current_nr_per_sex[sexes[cluster_centers]]+=1
            kk=1
        else:
            cluster_centers = given_centers
            kk=0

        distance_to_closest = np.amin(dmat[np.ix_(cluster_centers, np.arange(n))], axis=0)
        while kk<k:
            feasible_groups=np.where(current_nr_per_sex<nr_centers_per_sex)[0]
            feasible_points=np.where(np.isin(sexes,feasible_groups))[0]
            new_point=feasible_points[np.argmax(distance_to_closest[feasible_points])]
            current_nr_per_sex[sexes[new_point]] += 1
            cluster_centers = np.append(cluster_centers, new_point)
            distance_to_closest = np.amin(np.vstack((distance_to_closest, dmat[new_point, :])), axis=0)
            kk+=1

        cluster_centers=cluster_centers[given_centers.size:]

    return cluster_centers
########################################################################################################################




