import numpy as np
import scipy.sparse.csgraph

from sage.all import *
import sage.matroids.matroid
import sage.matroids.constructor



########################################################################################################################
### Implementation of the 3-approximation algorithm for the matroid center problem proposed by Chen et al.
### (Danny Z. Chen, Jian Li, Hongyu Liang, Haitao Wang. Matroid and Knapsack Center Problems. Algorithmica, 2016)
########################################################################################################################


class PartitionMatroid_adapted(sage.matroids.matroid.Matroid):
    '''Adaptation of the class PartitionMatroid as presented in the 'Sage Reference Manual: MatroidTheory' to the matroid
    required in the algorithm by Chen et al..

    partition ... list of lists specifying the partition of the groundset'''


    def __init__(self, partition):
        self.partition = partition
        E = set()
        for P in partition:
            E.update(P)
        self.E = frozenset(E)
    def groundset(self):
        return self.E
    def _rank(self, X):
        X2 = set(X)
        used_indices = set()
        rk = 0
        while len(X2) > 0:
            e = X2.pop()
            for i in range(len(self.partition)-1):
                if e in self.partition[i]:
                    if i not in used_indices:
                        used_indices.add(i)
                        rk = rk + 1
                    break
        return rk



class ValidCentersMatroid(sage.matroids.matroid.Matroid):
    '''Partition matroid encoding the constraints on the centers.

    sexes ... integer-vector of length n with entries in 0,...,m-1, where m is the number of groups
    nr_centers_per_sex ... integer-vector of length m with entries in 0,...,k and sum over entries equaling k'''


    def __init__(self, sexes,nr_centers_per_sex):
        self.n = sexes.size
        self.sexes=sexes
        self.m=nr_centers_per_sex.size
        self.nr_centers_per_sex=nr_centers_per_sex
        self.E = frozenset(np.arange(self.n))
    def groundset(self):
        return self.E
    def _rank(self, X):
        X2 = set(X)
        nr_elem_in_groups=np.zeros(self.m,dtype=int)
        while len(X2) > 0:
            e = X2.pop()
            nr_elem_in_groups[self.sexes[e]]+=1
        return np.sum(np.minimum(nr_elem_in_groups,self.nr_centers_per_sex))





def MatCenter_binary_search(dmat, sexes, nr_centers_per_sex):
    '''Implementation of the algorithm by Chen et al..

    *) Rather than testing all distance values as threshold as suggested by Chen et al., we implement binary search to
    look for the optimal value.
    *) There might be a faster way than running Floyd-Warshall for every distance value that we are testing, however,
    in our experiments the time for doing so is negligible (for n<=250, the execution of the first five commands within the
    while-loop never takes more than 0.02 seconds).

    INPUT:
    dmat ... distance matrix of size nxn
    sexes ... integer-vector of length n with entries in 0,...,m-1, where m is the number of groups
    nr_centers_per_sex ... integer-vector of length m with entries in 0,...,k and sum over entries equaling k

    RETURNS: approx. optimal centers'''


    n = dmat.shape[0]
    m = nr_centers_per_sex.size
    k = np.sum(nr_centers_per_sex)

    ConstrMatr =ValidCentersMatroid(sexes,nr_centers_per_sex)

    iu = np.triu_indices(n, 1)
    dval=np.sort(dmat[iu])

    best_cost=np.inf
    best_centers=np.array([],dtype=int)


    while dval.size>0:

        thelp=int(np.floor(dval.size/2))
        cand_dist=dval[thelp]

        dmat_c=dmat.copy()
        dmat_c[dmat_c>cand_dist]=0
        scipy.sparse.csgraph.floyd_warshall(dmat_c, directed=False, overwrite=True)

        CC=np.array([],dtype=int)
        VV=np.zeros(n)
        parti=[]
        parti2=np.arange(n)

        while np.sum(VV)<n:
            v=np.where(VV==0)[0][0]
            np.append(CC,v)
            temp=np.where(dmat_c[v, :] <= (cand_dist))[0]
            parti.append(temp)
            parti2=np.setdiff1d(parti2,temp)
            VV[dmat_c[v,:]<=(2*cand_dist)]=1

        parti.append(parti2)
        PartMatr=PartitionMatroid_adapted(parti)

        ttemp=np.array(list(ConstrMatr.intersection_unweighted(PartMatr)),dtype=int)

        if ttemp.size<CC.size:
            curr_cost=np.inf
            dval = dval[(thelp+1):]
        else:
            curr_cost = np.amax(np.amin(dmat[np.ix_(ttemp, np.arange(n))], axis=0))
            dval=dval[0:thelp]

        if curr_cost<best_cost:
            best_cost=curr_cost
            best_centers=ttemp


    for ell in np.arange(m):
        if np.sum(sexes[best_centers] == ell) < nr_centers_per_sex[ell]:
            toadd = nr_centers_per_sex[ell] - np.sum(sexes[best_centers] == ell)
            toadd_pot = np.setdiff1d(np.where(sexes == ell)[0], best_centers)
            if toadd_pot.size > toadd:
                best_centers = np.hstack((best_centers, toadd_pot[0:toadd]))
            else:
                best_centers = np.hstack((best_centers, toadd_pot))

    return best_centers



def MatCenter_binary_search_WithGivenCenters(dmat, sexes, nr_centers_per_sex, given_centers):
    '''Wrapper function that allows us to run the algorithm by Chen et al. with initially given centers.

    INPUT:
    dmat ... distance matrix of size nxn
    sexes ... integer-vector of length n with entries in 0,...,m-1, where m is the number of groups
    nr_centers_per_sex ... integer-vector of length m with entries in 0,...,k and sum over entries equaling k
    given_centers ... integer-vector with entries in 0,...,n-1

    RETURNS: approx. optimal centers'''


    m=nr_centers_per_sex.size
    sexesNEW=sexes.copy()
    sexesNEW[given_centers]=m
    nr_centers_per_sexNEW=np.hstack((nr_centers_per_sex,given_centers.size))

    return np.setdiff1d(MatCenter_binary_search(dmat, sexesNEW, nr_centers_per_sexNEW),given_centers)


