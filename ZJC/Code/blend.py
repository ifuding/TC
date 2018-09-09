import numpy as np
import pandas as pd
import time
from time import gmtime, strftime
from main import *
from sklearn import metrics
import lightgbm as lgb
from tensorflow.python.keras.models import load_model
from keras_train import DNN_Model

# len_train = 20905996
# len_valide = 20000001
# df = pd.read_pickle('MinMaxNormTrainValTest.pickle')

def load_val():
    valide_df = load_valide_data()
    # valide_df = df[len_train: len_train + len_valide]
    valide_id = valide_df['id'].values
    valide_data = valide_df[keras_train.USED_FEATURE_LIST].values.astype(DENSE_FEATURE_TYPE)
    valide_label = valide_df['is_attributed'].values.astype(np.uint8)
    del valide_df
    gc.collect()
    pos_cnt = valide_label.sum()
    neg_cnt = len(valide_label) - pos_cnt
    print ("valide type: {0} valide size: {1} valide data pos: {2} neg: {3}".format(
            valide_data.dtype, len(valide_data), pos_cnt, neg_cnt))
    return valide_data, valide_label, valide_id

def load_test():
    test_df = load_test_data()
    test_data = test_df[keras_train.USED_FEATURE_LIST].values.astype(DENSE_FEATURE_TYPE)
    test_id = test_df['click_id'].values #.astype(np.uint32)
    print ("test type {0}".format(test_data.dtype))
    del test_df
    gc.collect()
    return test_data, test_id

def lgb_pred(valide_data, valide_label):
    # load lightgbm model to predict
    bst = lgb.Booster(model_file= FLAGS.input_previous_model_path + '/model_098597.txt')
    lgb_pred = bst.predict(valide_data, num_iteration=FLAGS.best_iteration)
    score = metrics.roc_auc_score(valide_label, lgb_pred)
    print ("LightGbm AUC: {0}".format(score))
    return lgb_pred

def keras_pred(valide_data, valide_label):
    model = load_model(FLAGS.input_previous_model_path + '/model_0986303.h5')
    print (model.summary())
    y_pred = model.predict(DNN_Model.DNN_DataSet(None, valide_data), verbose=0, batch_size=10240)
    score = metrics.roc_auc_score(valide_label, y_pred)
    print ("Keras AUC: {0}".format(score))
    return y_pred

def blend(sub1, sub2):
    data_dir = "../../Data/"
    sub1 = pd.read_csv(data_dir + '/feature_scoring_vs_zeros/leaky_submission.csv', index_col = 'ID')
    sub2 = pd.read_csv(data_dir + 'submission_132.csv', index_col = 'ID')

    sub3 = pd.read_csv(data_dir + 'sub_2018_08_09_07_43_07.csv', index_col = 'ID')
    sub4 = pd.read_csv(data_dir + 'submission_134.csv', index_col = 'ID')
    target = 'target'
    sub1.rename(columns={'target': 't1'}, inplace = True)
    sub2.rename(columns={'target': 't2'}, inplace = True)
    sub3.rename(columns={'target': 't3'}, inplace = True)
    sub4.rename(columns={'target': 't4'}, inplace = True)

    blend1 = pd.concat([sub1, sub4], axis = 1)
    print(blend1.head())
    blend1['target'] = 0.8 * blend1['t1'] + 0.2 * blend1['t4']

    blend2 = pd.concat([sub2, sub3], axis = 1)
    blend2['target'] = 0.8 * blend2['t3'] + 0.2 * blend2['t2']

    blend3 = blend1
    blend3['target'] = 0.5 * blend3['target'] + 0.5 * blend2['target']

    # leak_target = pd.read_csv(data_dir + 'target_leaktarget_30_1.csv', index_col = 'ID')
    # leak_target = leak_target.loc[sub1.index, 'leak_target']
    # leak_target = leak_target[leak_target != 0]
    #blend 1
    # blend = sub1 #pd.merge(sub1, sub2, how='left', on='ID')
    # print (blend.info())
    # blend[target] = np.sqrt(blend[target + "_x"] * blend[target+'_y'])
    # blend[target] = (blend[target + "_x"] + blend[target+'_y'])/2
    # blend[target][leak_target.index] = leak_target
    # blend[target] = blend[target].clip(0+1e12, 1-1e12)

    time_label = strftime('_%Y_%m_%d_%H_%M_%S', gmtime())
    sub_name = data_dir + "sub" + time_label + ".csv"
    blend3[[target]].to_csv(sub_name)

def leak_blend():
    path = "../../Data/"
    sub = pd.read_csv(path + 'sub_2018_08_18_06_39_36.csv', index_col = 'ID')
    leak_target = pd.read_csv(path + 'add_featrure_set_target_leaktarget_38_3.csv', index_col = 'ID')['leak_target']
    sub.loc[leak_target[leak_target != 0].index, 'target'] = leak_target[leak_target != 0]

    time_label = strftime('_blend_leak_%Y_%m_%d_%H_%M_%S', gmtime())
    sub_name = path + "sub" + time_label + ".csv"
    sub.to_csv(sub_name)

def blend_tune(valide_label, sub1, sub2):
    sub1 = sub1.reshape((len(valide_label), -1))
    sub2 = sub2.reshape((len(valide_label), -1))
    print (sub1.shape)
    print (sub2.shape)
    blend1 = 0.97 * sub1 + 0.03 * sub2
    blend2 = np.sqrt((sub1 ** 0.45) * (sub2 ** 0.55))
    #blend = np.sqrt(sub1 * sub2)
    for i in range(30, 101, 1):
        r = float(i) / 100
        blend = (blend1 ** r) * (blend2 ** (1 - r))
        score = metrics.roc_auc_score(valide_label, blend)
        print ("r : {0} Blend AUC: {1}".format(r, score))

if __name__ == "__main__":
    if FLAGS.blend_tune:
        valide_data, valide_label, valide_id = load_val()
        k_pred = keras_pred(valide_data, valide_label)
        # df = pd.DataFrame()
        # df['id'] = valide_id
        # df['label'] = valide_label
        # df['re'] = k_pred
        # df = pd.read_pickle(path + 'valide_label_re.pickle')

        # pre_k_pred = np.load('../Data/TrainValNuniqueVarCumNextClickReversecum/k_pred.npy')
        # pre_l_pred = np.load('../Data/TrainValNuniqueVarCumNextClickReversecum/l_pred.npy')
        # pre_label = np.load('../Data/TrainValNuniqueVarCumNextClickReversecum/valide_label.npy')
        # pre_valide_id = np.load('../Data/TrainValNuniqueVarCumNextClickReversecum/valide_id.npy')
        # pre_df = pd.DataFrame()
        # pre_df['id'] = pre_valide_id
        # pre_df['label'] = pre_label
        # pre_df['re'] = np.sqrt(pre_k_pred.reshape((len(pre_label), -1)) * pre_l_pred.reshape((len(pre_label), -1)))
        # print (pre_df.head)
        # pre_df.to_pickle('../Data/TrainValNuniqueVarCumNextClickReversecum/valide_label_re.pickle')
        # pre_df = pd.read_pickle('../Data/TrainValNuniqueVarCumNextClickReversecum/valide_label_re.pickle')
        # df = pd.merge(df, pre_df, how = 'left', on = 'id')
        # print (df.info())
        # score = metrics.roc_auc_score(df['label_x'].values, df['re_y'].values)
        # print ("pre Blend AUC: {0}".format(score))
        # score = metrics.roc_auc_score(df['label_x'].values, np.sqrt(df['re_x'].values * df['re_y'].values))
        # print ("Blend AUC: {0}".format(score))
        # # l_pred = lgb_pred(valide_data, valide_label)
        # np.save(path + '/valide_id.npy', valide_id)
        # np.save('k_pred.npy', k_pred)
        # np.save('l_pred.npy', l_pred)
        # np.save('valide_label.npy', valide_label)
        # valide_label = np.load('valide_label.npy')
        # k_pred = np.load('k_pred.npy')
        # l_pred =  np.load('l_pred.npy')
        # blend_tune(valide_label, k_pred, l_pred)
    else:
        # test_data, test_id = load_test()
        # k_pred = keras_pred(test_data, test_id)
        # sub = pd.DataFrame()
        # sub['click_id'] = test_id
        # sub['is_attributed'] = k_pred
        leak_blend()
        # l_pred = lgb_pred(valide_data, valide_label)
