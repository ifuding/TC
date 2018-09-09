import pandas as pd
import time
import numpy as np
import gc
from feature_engineer import gen_features
from feature_engineer import timer
import keras_train
from nfold_train import nfold_train, models_eval
import tensorflow as tf
import os
import shutil
from lcc_sample import neg_sample
from sklearn import metrics
import lightgbm as lgb
from main import *


DENSE_FEATURE_TYPE = keras_train.DENSE_FEATURE_TYPE

def find_best_iteration_search(bst):
    """
    """
    valide_df = load_valide_data()
    valide_data = valide_df[keras_train.USED_FEATURE_LIST].values.astype(DENSE_FEATURE_TYPE)
    valide_label = valide_df['is_attributed'].values.astype(np.uint8)
    del valide_df
    gc.collect()
    if FLAGS.stacking:
        valide_data = gen_stacking_data(valide_data)
    pos_cnt = valide_label.sum()
    neg_cnt = len(valide_label) - pos_cnt
    print ("valide type: {0} valide size: {1} valide data pos: {2} neg: {3}".format(
            valide_data.dtype, len(valide_data), pos_cnt, neg_cnt))
    with timer("finding best iteration..."):
        search_iterations = [int(ii.strip()) for ii in FLAGS.search_iterations.split(',')]
        for i in range(search_iterations[0], search_iterations[1], search_iterations[2]):
            y_pred = bst.predict(valide_data, num_iteration=i)
            score = metrics.roc_auc_score(valide_label, y_pred)
            loss = metrics.log_loss(valide_label, y_pred)
            print ("Iteration: {0} AUC: {1} Logloss: {2}".format(i, score, loss))


def predict_test(bst):
    test_df = load_test_data()
    test_data = test_df[keras_train.USED_FEATURE_LIST].values.astype(DENSE_FEATURE_TYPE)
    test_id = test_df['click_id'].values #.astype(np.uint32)
    print ("test type {0}".format(test_data.dtype))
    del test_df
    gc.collect()
    if FLAGS.stacking:
        test_data = gen_stacking_data(test_data)
    with timer("predicting test data"):
        print('predicting test data...')
        sub_re = pd.DataFrame(test_id, columns = ['click_id'])
        sub_re['is_attributed'] = bst.predict(test_data, num_iteration=FLAGS.best_iteration)
        time_label = time.strftime('_%Y_%m_%d_%H_%M_%S', time.gmtime())
        sub_name = FLAGS.output_model_path + "sub" + time_label + ".csv"
        sub_re.to_csv(sub_name, index=False)


if __name__ == "__main__":
    # load model to predict
    bst = lgb.Booster(model_file= FLAGS.input_previous_model_path + '/model.txt')
    if FLAGS.search_best_iteration:
        find_best_iteration_search(bst)
    else:
        predict_test(bst)