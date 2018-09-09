#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as ss
# import dask.dataframe as dd
# from dask.multiprocessing import get
import concurrent.futures

def rank_INT(series, c=3.0/8, stochastic=True):
    """ Perform rank-based inverse normal transformation on pandas series.
        If stochastic is True ties are given rank randomly, otherwise ties will
        share the same value. NaN values are ignored.
        Args:
            param1 (pandas.Series):   Series of values to transform
            param2 (Optional[float]): Constand parameter (Bloms constant)
            param3 (Optional[bool]):  Whether to randomise rank of ties
        
        Returns:
            pandas.Series
    """
    # Check input
    assert(isinstance(series, pd.Series))
    assert(isinstance(c, float))
    assert(isinstance(stochastic, bool))

    # Set seed
    np.random.seed(123)

    # Take original series indexes
    orig_idx = series.index

    # Drop NaNs
    # series = series.loc[~pd.isnull(series)]

    # Get ranks
    if stochastic == True:
        # Shuffle by index
        series = series.loc[np.random.permutation(series.index)]
        # Get rank, ties are determined by their position in the series (hence
        # why we randomised the series)
        rank = ss.rankdata(series, method="ordinal")
    else:
        # Get rank, ties are averaged
        rank = ss.rankdata(series, method="average")
    
    transformed = rank_to_normal(rank, c, len(rank))
    # Convert numpy array back to series
    # rank = pd.Series(rank, index=series.index)

    # # Convert rank to normal distribution
    # transformed = rank_to_normal(rank, c, len(rank)) #rank.apply(rank_to_normal, c=c, n=len(rank))
    
    return pd.Series(transformed, index=series.index) #[orig_idx] #.values

def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2*c + 1)
    return ss.norm.ppf(x)

def rank_INT_DF(df):
    # ddata = dd.from_pandas(df.T, npartitions=8, sort = False)
    # return ddata.map_partitions(lambda df: df.apply(rank_INT, axis = 1)).compute().T
    MAX_WORKERS = 16
    cols = df.columns.values
    print(cols)
    col_ind_begin = 0
    col_len = cols.shape[0]
    while col_ind_begin < col_len:
        col_ind_end = min(col_ind_begin + MAX_WORKERS, col_len)
        with concurrent.futures.ThreadPoolExecutor(max_workers = MAX_WORKERS) as executor:
            future_predict = {executor.submit(rank_INT, df[cols[ind]]): ind for ind in range(col_ind_begin, col_ind_end)}
            for future in concurrent.futures.as_completed(future_predict):
                ind = future_predict[future]
                try:
                    df[cols[ind]] = future.result()
                except Exception as exc:
                    print('%dth feature normalize generate an exception: %s' % (ind, exc))
        col_ind_begin = col_ind_end
        if col_ind_begin % 100 == 0:
            print('Gen %d normalized features' % col_ind_begin)
    return df


def test():
    
    # Test
    s = pd.Series(np.random.randint(1, 10, 6), index=["a", "b", "c", "d", "e", "f"])
    print(s)
    res = rank_INT_DF(s)
    print(res)

    return 0

def test_df():
    
    # Test
    s = pd.DataFrame({'c0': np.random.randint(1, 10, 6), 'c1': np.random.randint(1, 10, 6)}, index=["a", "b", "c", "d", "e", "f"])
    print(s)
    res = rank_INT_DF(s)
    print(res)

    return 0

if __name__ == '__main__':

    test_df()