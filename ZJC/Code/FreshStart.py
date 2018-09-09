import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# print(os.listdir("../input"))

import lightgbm as lgb
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import mode, skew, kurtosis, entropy
from sklearn.ensemble import ExtraTreesRegressor

# import matplotlib.pyplot as plt
# import seaborn as sns

# import dask.dataframe as dd
# from dask.multiprocessing import get

from tqdm import tqdm, tqdm_notebook
tqdm.pandas(tqdm_notebook)
import concurrent.futures
import time
import pickle

from leak_cols import *
leak_list = LEAK_LIST

# Any results you write to the current directory are saved as output.
path = "../../Data/"
train = pd.read_csv(path + "train.csv", index_col = 'ID')
test = pd.read_csv(path + "test.csv", index_col = 'ID')

debug = False
if debug:
    train = train[:1000]
    test = test[:1000]
IsTrain = False

transact_cols = [f for f in train.columns if f not in ["ID", "target"]]
y = np.log1p(train["target"]).values

cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1',
       '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9',
       'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', 
       '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212',  '66ace2992',
       'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', 
       '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
       '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2',  '0572565c2',
       '190db8488',  'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'] 

from multiprocessing import Pool
CPU_CORES = 1
NZ_NUM = 3
def _get_leak(df, cols, search_ind, lag=0):
    """ To get leak value, we do following:
       1. Get string of all values after removing first two time steps
       2. For all rows we shift the row by two steps and again make a string
       3. Just find rows where string from 2 matches string from 1
       4. Get 1st time step of row in 3 (Currently, there is additional condition to only fetch value if we got exactly one match in step 3)"""
    f1 = [] #cols[:((lag+2) * -1)]
    f2 = [] #cols[(lag+2):]
    for ef in leak_list:
        f1 += ef[:((lag+2) * -1)]
        f2 += ef[(lag+2):]
    series_str = df[f2]
    nz = series_str.apply(lambda x: len(x[x!=0]), axis=1)
    series_str = series_str[nz >= NZ_NUM]
    series_str = series_str.apply(lambda x: "_".join(x.round(2).astype(str)), axis=1)
    series_str = series_str.drop_duplicates(keep = False) #[(~series_str.duplicated(keep = False)) | (df[cols[lag]] != 0)]
    series_shifted_str = df.loc[search_ind, f1].apply(lambda x: "_".join(x.round(2).astype(str)), axis=1)
    target_rows = series_shifted_str.progress_apply(lambda x: np.where(x == series_str.values)[0])
    # print(target_rows)
    # del series_str, series_shifted_str
    # target_vals = target_rows.apply(lambda x: df.loc[series_str.index[x[0]], cols[lag]] if len(x)==1 else 0)
    target_vals = target_rows.apply(lambda x: df.loc[series_str.index[x[0]], cols[lag]] if len(x) == 1 else 0)
        # if (len(x) > 0 and df.loc[series_str.index[x], cols[lag]].nunique() == 1) else 0)
    return target_vals, lag

def get_all_leak(df, cols=None, nlags=15):
    """
    We just recursively fetch target value for different lags
    """
    df =  df.copy()
#     print(df.head())
#     with Pool(processes=CPU_CORES) as p:
    if True:
        begin_ind = 0
        end_ind = 0
        leak_target = pd.Series(0, index = df.index)
        while begin_ind < nlags:
            end_ind = min(begin_ind + CPU_CORES, nlags)
            search_ind = (leak_target == 0)
            # print(search_ind)
            print('begin_ind: ', begin_ind, 'end_ind: ', end_ind, "search_ind_len: ", search_ind.sum())
#             res = [p.apply_async(_get_leak, args=(df, cols, search_ind, i)) for i in range(begin_ind, end_ind)]
#             for r in res:
#                 target_vals, lag = r.get()
# #                 print ('target_vale', target_vals.head())
#                 # leak_target[target_vals.index] = target_vals
#                 df['leak_target_' + str(lag)] = target_vals
            target_vals, lag = _get_leak(df, cols, search_ind, begin_ind)
            df['leak_target_' + str(lag)] = target_vals
            for i in range(begin_ind, end_ind):
                leak_target[leak_target == 0] = df.loc[leak_target == 0, 'leak_target_' + str(i)]
            leak_train = 0 #leak_target[train.index]
            leak_train_len = 0 #leak_train[leak_train != 0].shape[0]
            leak_test_len = 0 #leak_target[test.index][leak_target != 0].shape[0]
            leak_train_right_len = 0 #leak_train[leak_train.round(0) == train['target'].round(0)].shape[0]
            leak_train_right_ratio = 0 #leak_train_right_len / leak_train_len
            if IsTrain:
                leak_train = leak_target[train.index]
#                 print (leak_train.head())
                leak_train_len = leak_train[leak_train != 0].shape[0]
                leak_train_right_len = leak_train[leak_train.round(0) == train['target'].round(0)].shape[0]
                leak_train_right_ratio = leak_train_right_len / leak_train_len
            else:
                leak_test_len = leak_target[test.index][leak_target != 0].shape[0]
            print('Find leak in train and test: ', leak_train_len, leak_test_len, \
                "leak train right: ", leak_train_right_len, leak_train_right_ratio)
            begin_ind = end_ind
    # for i in range(nlags):
    #     df.loc[df['leak_target'] == 0, 'leak_target'] = df.loc[df['leak_target'] == 0, 'leak_target_' + str(i)]
    df['leak_target'] = leak_target
    return df

def get_pred(data, lag=2):
    d1 = data[FEATURES[:-lag]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2 = data[FEATURES[lag:]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2['pred'] = data[FEATURES[lag - 2]]
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    return d1.merge(d3, how='left', on='key').pred.fillna(0)

def get_all_pred(data, max_lag):
    target = pd.Series(index=data.index, data=np.zeros(data.shape[0]))
    for lag in range(2, max_lag + 1):
        pred = get_pred(data, lag)
        mask = (target == 0) & (pred != 0)
        target[mask] = pred[mask]
    return target

test["target"] = 0 #train["target"].mean()

# all_df = pd.concat([train[["ID", "target"] + cols], test[["ID", "target"]+ cols]]).reset_index(drop=True)
# all_df = pd.concat([train[["target"] + cols], test[["target"]+ cols]]) #.reset_index(drop=True)
# all_df.head()

NLAGS = 38 #Increasing this might help push score a bit
used_col = ["target"] + [col for cols in leak_list for col in cols]
print ('used_col length: ', len(used_col))
if IsTrain:
    all_df = get_all_leak(train[used_col], cols=cols, nlags=NLAGS)
else:
    all_df = get_all_leak(test[used_col], cols=cols, nlags=NLAGS)

if IsTrain:
    all_df[['target', 'leak_target']].to_csv(path + 'train_add_featrure_set_target_leaktarget_' + str(NLAGS) + "_" + str(NZ_NUM) + '.csv')
else:
    all_df[['target', 'leak_target']].to_csv(path + 'test_add_featrure_set_target_leaktarget_' + str(NLAGS) + "_" + str(NZ_NUM) + '.csv')
# with open(path + 'leak_target_' + str(NLAGS) + '.pickle', 'wb+') as handle:
#     pickle.dump(all_df[['target', 'leak_target']], handle, protocol=pickle.HIGHEST_PROTOCOL)

sub = pd.read_csv(path + 'sub_2018_08_13_03_19_33.csv', index_col = 'ID')
leak_target = all_df['leak_target'][test.index]
# print(leak_target)
sub.loc[leak_target[leak_target != 0].index, 'target'] = leak_target[leak_target != 0]

if not IsTrain:
    time_label = time.strftime('_%Y_%m_%d_%H_%M_%S', time.gmtime())
    sub.to_csv(path + 'leak_sub_' + str(NLAGS) + "_" + time_label + '.csv')
exit(0)

leaky_cols = ["leaked_target_"+str(i) for i in range(NLAGS)]
train = train.join(all_df.set_index("ID")[leaky_cols], on="ID", how="left")
test = test.join(all_df.set_index("ID")[leaky_cols], on="ID", how="left")

train[["target"]+leaky_cols].head(10)

train["nonzero_mean"] = train[transact_cols].apply(lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1)
test["nonzero_mean"] = test[transact_cols].apply(lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1)

#We start with 1st lag target and recusrsively fill zero's
train["compiled_leak"] = 0
test["compiled_leak"] = 0
for i in range(NLAGS):
    train.loc[train["compiled_leak"] == 0, "compiled_leak"] = train.loc[train["compiled_leak"] == 0, "leaked_target_"+str(i)]
    test.loc[test["compiled_leak"] == 0, "compiled_leak"] = test.loc[test["compiled_leak"] == 0, "leaked_target_"+str(i)]
    
print("Leak values found in train and test ", sum(train["compiled_leak"] > 0), sum(test["compiled_leak"] > 0))
print("% of correct leaks values in train ", sum(train["compiled_leak"] == train["target"])/sum(train["compiled_leak"] > 0))

# train.loc[train["compiled_leak"] == 0, "compiled_leak"] = train.loc[train["compiled_leak"] == 0, "nonzero_mean"]
# test.loc[test["compiled_leak"] == 0, "compiled_leak"] = test.loc[test["compiled_leak"] == 0, "nonzero_mean"]

from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y, np.log1p(train["compiled_leak"]).fillna(14.49)))

#submission
sub = test[["ID"]]
sub["target"] = test["compiled_leak"]
sub.to_csv(path + "baseline_submission_with_leaks.csv", index=False)