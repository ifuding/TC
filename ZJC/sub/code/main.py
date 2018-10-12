import pandas as pd
import time
import numpy as np
import tensorflow as tf
import pickle
import glob
import gc
import argparse
import sys

# flags = tf.app.flags
# flags.DEFINE_bool("debug", True, "Whether to load small data for debuging")
# flags.DEFINE_bool("predict_flat", False, "Whether to predict_flat")
# flags.DEFINE_string('input-training-data-path', "../data/", 'data dir override')
# flags.DEFINE_string('output-model-path', "../submit/", 'model dir override')
# flags.DEFINE_integer('densenet_nfold', 10, 'number of densenet nfold')
# flags.DEFINE_integer('dem_nfold', 5, 'number of dem nfold')
# flags.DEFINE_integer('densenet_ensemble_nfold', 1, 'number of ensemble models')
# flags.DEFINE_integer('dem_ensemble_nfold', 5, 'number of ensemble models')
# flags.DEFINE_integer('densenet_epochs', 1, 'number of Epochs')
# flags.DEFINE_integer('dem_epochs', 1, 'number of Epochs')
# flags.DEFINE_integer('densenet_batch_size', 256, 'Batch size')
# flags.DEFINE_integer('dem_batch_size', 64, 'Batch size')
# flags.DEFINE_integer('densenet_patience', 30, 'patience')
# flags.DEFINE_integer('dem_patience', 10, 'patience')
# flags.DEFINE_bool("aug_data", True, "aug_data")
# flags.DEFINE_string('blocks', "2,2", 'densenet blocks')
# flags.DEFINE_float("weight_decay", 1e-4, "weight_decay")
# flags.DEFINE_string('kernel_initializer', "glorot_normal", 'kernel_initializer')
# flags.DEFINE_integer('init_filters', 4, 'init_filters')
# flags.DEFINE_integer('growth_rate', 2, 'growth_rate')
# flags.DEFINE_float("reduction", 0.5, "reduction")
# flags.DEFINE_float("lr", 1e-3, "lr")
# flags.DEFINE_integer('init_stride', 2, 'init_stride')
# flags.DEFINE_integer('train_verbose', 1, 'train_verbose')
# flags.DEFINE_integer('img_flat_len', 1024, 'img_flat_len')
# flags.DEFINE_string('input-previous-model-path', "../../Data/", 'data dir override')
# flags.DEFINE_bool('load_img_model', False, 'load_img_model')
# flags.DEFINE_integer('cat_max', 0, 'cat_max')
# flags.DEFINE_string('zs_model_type', 'DEM', 'zs_model_type')

@staticmethod
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

default_parser = argparse.ArgumentParser()
default_parser.add_argument("--input-training-data-path", type=str, default=None)
default_parser.add_argument("--output-model-path", type=str, default=None)
default_parser.add_argument("--input-previous-model-path", type=str, default=None)
default_parser.add_argument("--debug", type=str2bool.__func__, default=True)
default_parser.add_argument("--predict_flat", type=str2bool.__func__, default=None)
default_parser.add_argument("--train_verbose", type=int, default=None)
default_parser.add_argument("--load_img_model", type=str2bool.__func__, default=False)
## DenseNet args
default_parser.add_argument("--densenet_nfold", type=int, default=None)
default_parser.add_argument("--densenet_ensemble_nfold", type=int, default=None)
default_parser.add_argument("--densenet_epochs", type=int, default=None)
default_parser.add_argument("--densenet_batch_size", type=int, default=None)
default_parser.add_argument("--densenet_patience", type=int, default=None)
default_parser.add_argument("--blocks", type=str, default=None)
default_parser.add_argument("--weight_decay", type=float, default=None)
default_parser.add_argument("--init_filters", type=int, default=None)
default_parser.add_argument("--growth_rate", type=int, default=None)
default_parser.add_argument("--reduction", type=float, default=None)
default_parser.add_argument("--lr", type=float, default=None)
default_parser.add_argument("--init_stride", type=int, default=None)
default_parser.add_argument("--cat_max", type=int, default=None)
default_parser.add_argument("--aug_data", type=str2bool.__func__, default=True)
default_parser.add_argument("--kernel_initializer", type=str, default=None)
## DEM args
default_parser.add_argument("--dem_nfold", type=int, default=None)
default_parser.add_argument("--dem_ensemble_nfold", type=int, default=None)
default_parser.add_argument("--dem_epochs", type=int, default=None)
default_parser.add_argument("--dem_batch_size", type=int, default=None)
default_parser.add_argument("--dem_patience", type=int, default=None)
default_parser.add_argument("--img_flat_len", type=int, default=None)
default_parser.add_argument("--zs_model_type", type=str, default=None)
## Data Augmentation args
default_parser.add_argument("--rotation_range", type=int, default=0)
default_parser.add_argument("--shear_range", type=float, default=0.)
default_parser.add_argument("--zoom_range", type=float, default=0.)
default_parser.add_argument("--horizontal_flip", type=str2bool.__func__, default=False)

# FLAGS = flags.FLAGS
(FLAGS, unknown) = default_parser.parse_known_args(sys.argv)
path = FLAGS.input_training_data_path
model_path = FLAGS.input_previous_model_path

import sklearn
from keras.preprocessing import image
from DenseNet import DenseNet
from DEM import DEM
from sklearn.model_selection import KFold
from utils import model_eval, models_eval, multi_models_vote, extract_array_from_series
import shutil
import os
from tqdm import tqdm, tqdm_notebook
tqdm.pandas(tqdm_notebook)

def read_class_emb(class_emb_path):
    class_emb = pd.read_csv(class_emb_path, index_col = 0, sep = ' ', header = None)
    class_emb.index.name = 'class_name'
    class_emb = class_emb.apply(lambda s: np.array([float(x) for x in s])[:300], axis = 1)
    return class_emb

def load_data():
    print("\nData Load Stage")
    with open(path + 'round2_class_id_emb_attr.pkl', 'rb') as handle:
        class_id_emb_attr = pickle.load(handle)
    with open(path + '/round1_train_img_part0.pkl', 'rb') as handle:
        round1_train_img_part0 = pickle.load(handle)
    with open(path + '/round1_train_img_part1.pkl', 'rb') as handle:
        round1_train_img_part1 = pickle.load(handle)
    with open(path + '/round2_train_img.pkl', 'rb') as handle:
        round2_train_img = pickle.load(handle)
    with open(path + '/round2_test_img.pkl', 'rb') as handle:
        test_data = pickle.load(handle)
    round2_class_id = ['ZJL' + str(i) for i in range(296, 501)]
    round2_train_class_id = round2_train_img.class_id.unique()
    train_data = pd.concat([round1_train_img_part0, round1_train_img_part1, round2_train_img], axis = 0, sort = False)
    del round1_train_img_part0, round1_train_img_part1, round2_train_img
    gc.collect()
    train_data = train_data.merge(class_id_emb_attr, how = 'left', on = 'class_id')
    if FLAGS.debug:
        train_data = train_data.iloc[:200]
        test_data = test_data.iloc[:200]

    # glove_emb = read_class_emb(path + '/DatasetB/class_wordembeddings.txt')
    # fasttext_emb =  read_class_emb(path + '/External/class_wordembeddings_fasttext')

    # class_id_to_name = pd.read_csv(path + '/DatasetB/label_list.txt', 
    #                             index_col = 'class_name', sep = '\t', header = None, names = ['class_id', 'class_name'])

    # attr_list = pd.read_csv(path + '/DatasetB/attribute_list.txt', index_col = 0, sep = '\t', header = None)

    # attributes_per_class = pd.read_csv(path + '/DatasetB/attributes_per_class.txt', 
    #                                 index_col = 0, sep = '\t', header = None)
    # attributes_per_class.index.name = 'class_id'
    # attributes_per_class = attributes_per_class.apply(lambda s: np.array([float(x) for x in s]), axis = 1)

    # class_id_emb_attr = class_id_to_name.copy()
    # class_id_emb_attr['emb_glove'] = glove_emb
    # class_id_emb_attr['emb_fasttext'] = fasttext_emb
    # class_id_emb_attr['emb'] = class_id_emb_attr.apply(lambda s: np.hstack([s['emb_glove'], s['emb_fasttext']]), axis = 1)
    # class_id_emb_attr.reset_index(inplace = True)
    # class_id_emb_attr.set_index('class_id', inplace = True)
    # class_id_emb_attr['attr'] = attributes_per_class
    # class_id_emb_attr.reset_index(inplace = True)
    # print ('Load class_id_emb_attr Done')
    
    # def read_image_data(img_id_path, imag_path, cols, debug):
    #     img_data = pd.read_csv(img_id_path, sep = '\t', header = None, names = cols)
    #     if FLAGS.debug:
    #         img_data = img_data[:2000]
    #     img_data['img'] = img_data['img_id'].progress_apply(lambda id: image.img_to_array(image.load_img(imag_path + id)))
    #     return img_data
    # print ('Load setA_train_data ----')
    # setA_train_data = read_image_data(img_id_path = path + '/DatasetA/train.txt', imag_path = path + '/DatasetA/train/', 
    #             cols = ['img_id', 'class_id'], debug = FLAGS.debug)
    # print ('Load setB_train_data ----')
    # setB_train_data = read_image_data(img_id_path = path + '/DatasetB/train.txt', imag_path = path + '/DatasetB/train/', 
    #             cols = ['img_id', 'class_id'], debug = FLAGS.debug)
    # train_data = setA_train_data.append(setB_train_data)
    # del setA_train_data, setB_train_data
    # train_data = train_data.merge(class_id_emb_attr, how = 'left', on = 'class_id')
    
    # print ('Load test_data ----')
    # test_data = read_image_data(img_id_path = path + '/DatasetB/image.txt', imag_path = path + '/DatasetB/test/', 
    #             cols = ['img_id'], debug = FLAGS.debug)

    return train_data, test_data, class_id_emb_attr, round2_class_id, round2_train_class_id

# train_data, test_data, class_id_emb_attr = load_data()

def train_img_classifier(train_data, flags):
    print("Over all training size:")
    print(train_data.shape)

    fold = flags.densenet_nfold
    ensemble_nfold = flags.densenet_ensemble_nfold
    kf = KFold(n_splits=fold, shuffle=True, random_state = 100)
    num_fold = 0
    models = []
    model_type = 'DenseNet'
    scores = []
    category = train_data['class_id'].unique()
    category_dict = dict((category[i], i) for i in range(category.shape[0]))
    if flags.load_img_model:
        img_model = DenseNet(scores = scores, cat_max = flags.cat_max, flags = flags, model_type = model_type).model
        model_file_name = glob.glob(model_path + '/imgmodel_*.h5')[0]
        print ('Model file name: ', model_file_name)
        img_model.load_weights(model_file_name)
        return (img_model, model_type)

    OneHotEncoder = sklearn.preprocessing.OneHotEncoder()
    train_target = train_data['class_id'].apply(lambda id: category_dict[id]).values
    train_target = OneHotEncoder.fit_transform(np.reshape(train_target, (-1, 1))).toarray()
    for train_index, test_index in kf.split(train_data):
        train_part_df = train_data.iloc[train_index]
        validate_part_df = train_data.iloc[test_index]

        train_part_label = train_target[train_index]
        validate_part_label = train_target[test_index]

        cat_max = train_part_df.class_id.unique().shape[0]
        print('\nfold: %d th train :-)' % (num_fold))
        print('Train size: {} Valide size: {}'.format(train_part_df.shape[0], validate_part_df.shape[0]))
        model = DenseNet(scores = scores, cat_max = cat_max, flags = flags, model_type = model_type)
        if num_fold == 0:
            print(model.model.summary())
        model.train(train_part_df, train_part_label, validate_part_df, validate_part_label)
        models.append((model.model, model_type))
        num_fold += 1
        if num_fold == ensemble_nfold:
            break
    return models[0]
# img_model = train_img_classifier(train_data, flags = FLAGS)

def train_zs_model(train_data, class_id_emb_attr, flags, img_flat_len,
                   round1_class_id = None,
                   round2_class_id = None,
                   img_model = None):
    print("Over all training size:")
    print(train_data.shape)

    fold = flags.dem_nfold
    ensemble_nfold = flags.dem_ensemble_nfold
    kf = KFold(n_splits=fold, shuffle=True, random_state = 100)
    num_fold = 0
    models = []
    model_type = FLAGS.zs_model_type
    scores = []
    classes = train_data.class_id.unique()

    for train_index, test_index in kf.split(classes):
        print ('Fold', num_fold, 'training...')
        seen_class = classes[train_index]
        unseen_class = classes[test_index]
        
        train_part_df = train_data[train_data.class_id.isin(seen_class)]
        validate_part_df = train_data[train_data.class_id.isin(unseen_class)]

        seen_round1_id = np.intersect1d(seen_class, round1_class_id)
        seen_round2_id = np.intersect1d(seen_class, round2_class_id)
        unseen_round1_id = np.intersect1d(unseen_class, round1_class_id)
        unseen_round2_id = np.intersect1d(unseen_class, round2_class_id)
        print ('Seen unseen Classes: ', seen_class.shape[0], unseen_class.shape[0])
        print ('Seen round1, round2: ', seen_round1_id.shape[0], seen_round2_id.shape[0])
        print ('Unseen round1, round2: ', unseen_round1_id.shape[0], unseen_round2_id.shape[0])

        zs_model = DEM(scores = scores, flags = flags, model_type = model_type, 
                    seen_class = seen_class, img_flat_len = img_flat_len, 
                    unseen_class = unseen_class,
                    class_id_emb_attr = class_id_emb_attr,
                    unseen_round1_id = unseen_round1_id,
                    unseen_round2_id = unseen_round2_id,
                    img_model = img_model)
        if num_fold == 0:
            print (zs_model.model.summary())
        zs_model.train(train_part_df, validate_part_df)
        models.append((zs_model.model, model_type))
        num_fold += 1
        if num_fold == ensemble_nfold:
            break
    score_df = pd.concat(scores, sort = False)
    agg_dict = {}
    statistic_columns = ['mean', 'median', 'max', 'min', 'std']
    for c in score_df.columns:
        agg_dict[c] = statistic_columns
    avg_score_df = score_df.agg(agg_dict)
    print (avg_score_df)
    return models, avg_score_df

# train_data['target'] = list(model_eval(img_model[0], img_model[1], train_data))
# test_data['target'] = list(model_eval(img_model[0], img_model[1], test_data))
# zs_models = train_zs_model(train_data, class_id_emb_attr, flags = FLAGS, img_flat_len = 128)

def predict_flat(img_model, train_data, test_data):
    time_label = time.strftime('%Y%m%d_%H%M%S')
    tmp_model_dir = "./model_sub/"
    if not os.path.isdir(tmp_model_dir):
        os.makedirs(tmp_model_dir, exist_ok=True)
    with open(tmp_model_dir + '/train_data_img_flat_' + time_label + '.pickle', 'wb') as handle:
        pickle.dump(extract_array_from_series(train_data['target']), handle)
    with open(tmp_model_dir + '/train_data_pred_img_class_' + time_label + '.pickle', 'wb') as handle:
        pickle.dump(extract_array_from_series(train_data['pred_img_class']), handle)
    with open(tmp_model_dir + '/test_data_img_flat_' + time_label + '.pickle', 'wb') as handle:
        pickle.dump(extract_array_from_series(test_data['target']), handle)
    with open(tmp_model_dir + '/test_data_pred_img_class_' + time_label + '.pickle', 'wb') as handle:
        pickle.dump(extract_array_from_series(test_data['pred_img_class']), handle)

    if not os.path.isdir(FLAGS.output_model_path):
        os.makedirs(FLAGS.output_model_path, exist_ok=True)
    for fileName in os.listdir(tmp_model_dir):
        dst_file = os.path.join(FLAGS.output_model_path, fileName)
        if os.path.exists(dst_file):
            os.remove(dst_file)
        shutil.move(os.path.join(tmp_model_dir, fileName), FLAGS.output_model_path)

def sub(models, train_data, test_data, class_id_emb_attr, img_model, score_df):
    train_id = train_data['class_id'].unique()
    test_img_feature_map = None
    if FLAGS.zs_model_type != 'DEM_AUG':
        test_img_feature_map = extract_array_from_series(test_data['target'])
    preds = multi_models_vote(models = models, eval_df = test_data, \
            cand_class_id_emb_attr = class_id_emb_attr[~class_id_emb_attr['class_id'].isin(train_id)], \
            img_feature_map = test_img_feature_map)
    sub = pd.DataFrame(preds, index = test_data['img_id'])
    time_label = time.strftime('%Y%m%d_%H%M%S')
    tmp_model_dir = "./model_sub/"
    if not os.path.isdir(tmp_model_dir):
        os.makedirs(tmp_model_dir, exist_ok=True)
    sub_name = tmp_model_dir + "/submit_"+ time_label + ".txt"
    sub.to_csv(sub_name, header = False, sep = '\t')
    score_df.to_csv(tmp_model_dir + '/scores.tsv')    

    model_name = tmp_model_dir + "imgmodel_" + time_label + ".h5"
    img_model[0].save(model_name)
    for i, model in enumerate(models):
        model_name = tmp_model_dir + "zsmodel_" + str(i) + time_label + ".h5"
        model[0].save(model_name)

    if not os.path.isdir(FLAGS.output_model_path):
        os.makedirs(FLAGS.output_model_path, exist_ok=True)
    for fileName in os.listdir(tmp_model_dir):
        dst_file = os.path.join(FLAGS.output_model_path, fileName)
        if os.path.exists(dst_file):
            os.remove(dst_file)
        shutil.move(os.path.join(tmp_model_dir, fileName), FLAGS.output_model_path)

# sub(models = zs_models, train_data = train_data, test_data = test_data, class_id_emb_attr = class_id_emb_attr)

if __name__ == "__main__":
    train_data, test_data, class_id_emb_attr, round2_class_id, round2_train_class_id = load_data()
    img_model = train_img_classifier(train_data, flags = FLAGS)
    if FLAGS.zs_model_type != 'DEM_AUG':
        train_preds = model_eval(img_model[0], img_model[1], train_data, verbose = FLAGS.train_verbose)
        test_preds = model_eval(img_model[0], img_model[1], test_data, verbose = FLAGS.train_verbose)
        train_data['target'] = list(train_preds[0])
        test_data['target'] = list(test_preds[0])
        train_data['pred_img_class'] = list(train_preds[1])
        test_data['pred_img_class'] = list(test_preds[1])
    if FLAGS.predict_flat:
        predict_flat(img_model, train_data, test_data)
    else:
        round1_class_id = list(set(train_data.class_id.unique()) - set(round2_class_id))
        zs_models, score_df = train_zs_model(train_data, #[train_data.class_id.isin(round2_class_id)], 
                class_id_emb_attr = class_id_emb_attr, #[class_id_emb_attr.class_id.isin(round2_class_id)], 
                flags = FLAGS, 
                img_flat_len = FLAGS.img_flat_len,
                round1_class_id = round1_class_id,
                round2_class_id = round2_class_id,
                img_model = img_model)
        cand_class_id_emb_attr = class_id_emb_attr[class_id_emb_attr.class_id.isin(round2_class_id)]
        sub(models = zs_models, train_data = train_data, test_data = test_data, class_id_emb_attr = cand_class_id_emb_attr, \
            img_model = img_model, score_df = score_df)