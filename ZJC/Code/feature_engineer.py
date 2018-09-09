import pandas as pd
import numpy as numpy
from contextlib import contextmanager
import gc
import time

@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print('[{0}] done in {1} s\n'.format(name, time.time() - t0))


def gen_features(train_df):
    """
    """
    with timer("grouping by ip-day-hour combination"):
        print('grouping by ip-day-hour combination...')
        gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
        train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
        del gp
        gc.collect()

    with timer("grouping by ip-app combination"):  
        print('grouping by ip-app combination...')
        gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
        train_df = train_df.merge(gp, on=['ip','app'], how='left')
        del gp
        gc.collect()

    with timer("grouping by ip-app-os combination"):
        print('grouping by ip-app-os combination...')
        gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
        train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
        del gp
        gc.collect()

    with timer("grouping by : ip_day_chl_var_hour"):
        # Adding features with var and mean hour (inspired from nuhsikander's script)
        print('grouping by : ip_day_chl_var_hour')
        gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
        train_df = train_df.merge(gp, on=['ip','day','channel'], how='left')
        del gp
        gc.collect()

    with timer("grouping by : ip_app_os_var_hour"):
        print('grouping by : ip_app_os_var_hour')
        gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
        train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
        del gp
        gc.collect()

    with timer("grouping by : ip_app_channel_var_day"):
        print('grouping by : ip_app_channel_var_day')
        gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
        train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
        del gp
        gc.collect()

    with timer("grouping by : ip_app_chl_mean_hour"):
        print('grouping by : ip_app_chl_mean_hour')
        gp = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
        print("merging...")
        train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
        del gp
        gc.collect()

    train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
    train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')
    train_df['ip_tchan_count'] = train_df['ip_tchan_count'].astype('float32')
    train_df['ip_app_os_var'] = train_df['ip_app_os_var'].astype('float32')
    train_df['ip_app_channel_mean_hour'] = train_df['ip_app_channel_mean_hour'].astype('float32')
    train_df['ip_app_channel_var_day'] = train_df['ip_app_channel_var_day'].astype('float32')
    train_df['is_attributed'] = train_df['is_attributed'].astype('float16')
    train_df['click_id'] = train_df['click_id'].astype('float32')

    return train_df