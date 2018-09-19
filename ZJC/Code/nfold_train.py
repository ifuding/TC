from sklearn.model_selection import KFold
from lgb import lgbm_train
# import xgboost as xgb
# from functools import reduce
import numpy as np
from keras_train import DNN_Model, VAE_Model
import keras_train
# import gensim
# from RCNN_Keras import get_word2vec, RCNN_Model
# from RNN_Keras import RNN_Model
from tensorflow.python.keras.models import Model
# from xgb import xgb_train
import pandas as pd
from sklearn import metrics
from tensorflow.python.keras.applications import vgg16

# RNN_PARAMS
RCNN_HIDDEN_UNIT = [128, 64]

def extract_array_from_series(s):
    return np.asarray(list(s))

def nfold_train(train_data, train_label, model_types = None,
            stacking = False, valide_data = None, valide_label = None,
            test_data = None, train_weight = None, valide_weight = None,
            flags = None ,tokenizer = None, scores = None, emb_weight = None, cat_max = None, leak_target = None):
    """
    nfold Training
    """
    print("Over all training size:")
    print(train_data.shape)
    print("Over all label size:")
    print(train_label.shape)

    fold = flags.nfold
    kf = KFold(n_splits=fold, shuffle=True)
    # wv_model = gensim.models.Word2Vec.load("wv_model_norm.gensim")
    stacking = flags.stacking
    stacking_data = pd.Series([np.zeros(1)] * train_data.shape[0])
    stacking_label = pd.Series([np.zeros(1)] * train_data.shape[0])
    test_preds = None
    num_fold = 0
    models = []
    losses = []
    train_part_img_id = []
    validate_part_img_id = []
    for train_index, test_index in kf.split(train_data):
        # print(test_index[:100])
        # exit(0)
        if valide_label is None:
            train_img = extract_array_from_series(train_data['img'])
            train_img = vgg16.preprocess_input(train_img)

            train_part = train_img[train_index]
            train_part_label = train_label[train_index]
            validate_part = train_img[test_index]
            validate_part_label = train_label[test_index]

            train_part_img_id.append(train_data.iloc[train_index].img_id)
            validate_part_img_id.append(train_data.iloc[test_index].img_id)

        print('\nfold: %d th train :-)' % (num_fold))
        print('Train size: {} Valide size: {}'.format(train_part_label.shape[0], validate_part_label.shape[0]))
        print ('Train target nunique: ', np.unique(np.argwhere(train_part_label == 1)[:, 1]).shape[0], 
           'Validate target nuique: ', np.unique(np.argwhere(validate_part_label == 1)[:, 1]).shape[0])
        onefold_models = []
        for model_type in model_types:
            if model_type == 'k' or model_type == 'r':
                # with tf.device('/cpu:0'):
                model = DNN_Model(scores = scores, cat_max = train_part_label.shape[1], flags = flags, emb_weight = emb_weight, model_type = model_type)
                if num_fold == 0:
                    print(model.model.summary())
                model.train(train_part, train_part_label, validate_part, validate_part_label)
                onefold_models.append((model, model_type))

            if stacking:
                flat_model = Model(inputs = model.model.inputs, outputs = model.model.get_layer(name = 'avg_pool').output)
                stacking_data[test_index] = list(flat_model.predict(validate_part))
                stacking_label[test_index] = list(model.predict(validate_part))
        models.append(onefold_models[0])
        num_fold += 1
        if num_fold == flags.ensemble_nfold:
            break
    # if stacking:
    #     test_preds /= flags.ensemble_nfold
    #     test_data = np.c_[test_data, test_preds]
    return models, stacking_data, stacking_label, test_preds, train_part_img_id, validate_part_img_id


def model_eval(model, model_type, data_frame):
    """
    """
    if model_type == 'l':
        preds = model.predict(data_frame[keras_train.USED_FEATURE_LIST].values, num_iteration=model.best_iteration)
    elif model_type == 'k' or model_type == 'LR' or model_type == 'DNN' or model_type == 'rcnn' \
        or model_type == 'r' or model_type == 'cnn':
        preds = model.predict(data_frame, verbose = 2)
    elif model_type == 'v':
        preds = model.predict(data_frame[keras_train.USED_FEATURE_LIST].values, verbose = 2)
        return preds
    elif model_type == 't':
        print("ToDO")
    elif model_type == 'x':
        preds = model.predict(xgb.DMatrix(data_frame), ntree_limit=model.best_ntree_limit)
    return preds #.reshape((data_frame.shape[0], ))

def models_eval(models, data):
    preds = None
    for (model, model_type) in models:
        pred = model_eval(model, model_type, data)
        if preds is None:
            preds = pred.copy()
        else:
            preds += pred
    preds /= len(models)
    return preds