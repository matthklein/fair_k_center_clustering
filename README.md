# fair_k_center_clustering

Code for our paper "Fair k-Center Clustering for Data Summarization" (https://arxiv.org/abs/1901.08628).

To try it out and reproduce the boxplots (based on 10 runs) of the experiments of Figures 4 to 6 on artificial data, simply run

```
python experiments_artificial_data.py 
```

If you want to obtain the boxplots based on 50 runs, say, then run

```
python experiments_artificial_data.py 50
```

Similarly, in order to reproduce the boxplots of the experiments of Figures 5 and 6 on the Adult data set, run

```
python experiments_adult_data_set.py 50
```

If you want to compare our algorithm to the algorithm for the matroid center problem by Chen et al. (https://arxiv.org/abs/1301.0745), you need to have SageMath (http://www.sagemath.org/) installed on your system. Then simply run

```
sage -python experiments_matroid_center.py 50
```

The code has been tested with the following software versions:
- Python 2.7.10
- Numpy 1.16.2
- Scipy 1.1.0
- Pandas 0.23.0
- SageMath 8.2
