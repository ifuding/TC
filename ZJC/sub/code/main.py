import pandas as pd
import time
import numpy as np
import tensorflow as tf
import pickle
import glob
import gc
import argparse
import sys
# import autokeras as ak

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
default_parser.add_argument("--predict_prob", type=str2bool.__func__, default=None)
default_parser.add_argument("--train_verbose", type=int, default=None)
default_parser.add_argument("--load_img_model", type=str2bool.__func__, default=False)
default_parser.add_argument("--load_zs_model", type=str2bool.__func__, default=False)
default_parser.add_argument("--only_emb", type=str2bool.__func__, default=False)
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
default_parser.add_argument("--pixel", type=int, default=None)
## DEM args
default_parser.add_argument("--dem_nfold", type=int, default=None)
default_parser.add_argument("--dem_ensemble_nfold", type=int, default=None)
default_parser.add_argument("--dem_epochs", type=int, default=None)
default_parser.add_argument("--dem_batch_size", type=int, default=None)
default_parser.add_argument("--dem_patience", type=int, default=None)
default_parser.add_argument("--img_flat_len", type=int, default=None)
default_parser.add_argument("--zs_model_type", type=str, default=None)
default_parser.add_argument("--wv_len", type=int, default=None)
default_parser.add_argument("--attr_len", type=int, default=None)
default_parser.add_argument("--attr_emb_len", type=int, default=None)
default_parser.add_argument("--attr_emb_transform", type=str, default=None)
default_parser.add_argument("--res_dem_epochs", type=int, default=None)
default_parser.add_argument("--res_dem_nfold", type=int, default=None)
default_parser.add_argument("--only_use_round2", type=str2bool.__func__, default=False)
## Data Augmentation args
default_parser.add_argument("--rotation_range", type=int, default=0)
default_parser.add_argument("--shear_range", type=float, default=0.)
default_parser.add_argument("--zoom_range", type=float, default=0.)
default_parser.add_argument("--horizontal_flip", type=str2bool.__func__, default=False)
default_parser.add_argument("--TTA", type=int, default=None)
default_parser.add_argument("--neg_aug", type=int, default=None)
default_parser.add_argument("--c2c_neg_cnt", type=int, default=None)
## ENAS
default_parser.add_argument("--enas", type=str2bool.__func__, default=False)
default_parser.add_argument("--enas_fold", type=int, default=None)
default_parser.add_argument("--enas_time", type=int, default=None)
## FastText
default_parser.add_argument("--train_ft", type=str2bool.__func__, default=False)
default_parser.add_argument("--ft_model", type=str, default='skipgram')
default_parser.add_argument("--ft_size", type=int, default=None)
default_parser.add_argument("--ft_threads", type=int, default=None)
default_parser.add_argument("--ft_iter", type=int, default=None)
default_parser.add_argument("--ft_verbose", type=int, default=None)
default_parser.add_argument("--ft_lrUpdateRate", type=int, default=None)
default_parser.add_argument("--ft_min_count", type=int, default=None)


# FLAGS = flags.FLAGS
(FLAGS, unknown) = default_parser.parse_known_args(sys.argv)
path = FLAGS.input_training_data_path
model_path = FLAGS.input_previous_model_path

import sklearn
from tensorflow.python.keras.models import Model, load_model
from DenseNet import DenseNet
from DEM import DEM
from sklearn.model_selection import KFold
from utils import model_eval, models_eval, multi_models_vote, extract_array_from_series, preprocess_img
import shutil
import os
from tqdm import tqdm, tqdm_notebook
tqdm.pandas(tqdm_notebook)
import gensim
import os
from gensim.models.wrappers import FastText
import tempfile

def read_class_emb(class_emb_path):
    class_emb = pd.read_csv(class_emb_path, index_col = 0, sep = ' ', header = None)
    class_emb.index.name = 'class_name'
    class_emb = class_emb.apply(lambda s: np.array([float(x) for x in s])[:300], axis = 1)
    return class_emb

def encode_attr():
    attr1_list = pd.read_csv(path + '/DatasetA/attribute_list.txt', index_col = 0, sep = '\t', header = None)
    attr2_list = pd.read_csv(path + '/semifinal_image_phase2/attribute_list.txt', index_col = 0, sep = '\t', header = None)
    attr_dict ={}
    round2_class_id = ['ZJL' + str(i) for i in range(296, 521)]
    round1_class_id = list(set(class_id_emb_attr.class_id.unique()) - set(round2_class_id))
    round1_class_id_emb_attr = class_id_emb_attr[class_id_emb_attr.class_id.isin(round1_class_id)]
    round2_class_id_emb_attr = class_id_emb_attr[class_id_emb_attr.class_id.isin(round2_class_id)]
    round1_attr_value = extract_array_from_series(round1_class_id_emb_attr['attr'])
    round2_attr_value = extract_array_from_series(round2_class_id_emb_attr['attr'])
    round1_attrs = attr1_list[1].values
    round2_attrs = attr2_list[1].values
    # round1_attr_to_ind = {}
    # round2_attr_to_ind = {}
    for i, attr in enumerate(list(attr1_list[1].values)):
        if attr not in attr_dict:
            attr_dict[attr] = set()
        attr_dict[attr].update(set(round1_attr_value[:, i]))
    #     round1_attr_to_ind[attr] = i
        
    for i, attr in enumerate(list(attr2_list[1].values)):
        if attr not in attr_dict:
            attr_dict[attr] = set()
        attr_dict[attr].update(set(round2_attr_value[:, i]))
    #     round2_attr_to_ind[attr] = i
        
    consant_attr = [attr for attr in attr_dict if len(attr_dict[attr]) == 1 ]

    round1_attr_df = pd.DataFrame(round1_attr_value, columns = round1_attrs)
    round2_attr_df = pd.DataFrame(round2_attr_value, columns = round2_attrs)
    attr_df = pd.concat([round1_attr_df, round2_attr_df], sort = False).fillna(-1).drop(columns = consant_attr).astype('str')
    attr_df = attr_df.apply(lambda s: s.name + '_' + s, axis = 0)
    # attr_df.values
    attr_value_dict = dict((v, i) for i, v in enumerate(set(attr_df.values.flatten())))
    encoded_attr_df = attr_df.apply(lambda s: [attr_value_dict[v] for v in s], axis = 0)
    class_id_emb_attr['attr'] = list(encoded_attr_df.values)
    encode_attr_min = encoded_attr_df.min().min()
    encode_attr_max = encoded_attr_df.max().max()
    encode_attr_min, encode_attr_max

def load_data():
    print("\nData Load Stage")
    with open(path + '/round2B_class_id_emb_attr.pkl', 'rb') as handle:
        class_id_emb_attr = pickle.load(handle)
        # class_id_emb_attr.drop(columns = ['emb_glove', 'emb_fasttext', 'emb_glove_crawl', 'emb_glove_crawl_42B'])
    with open(path + '/round1A_train_img.pkl', 'rb') as handle:
        round1_train_img_part0 = pickle.load(handle)
    with open(path + '/round1B_train_img.pkl', 'rb') as handle:
        round1_train_img_part1 = pickle.load(handle)
    with open(path + '/round2A_train_img.pkl', 'rb') as handle:
        round2_train_img = pickle.load(handle)
    with open(path + '/round2B_train_img.pkl', 'rb') as handle:
        round2B_train_img = pickle.load(handle)
    with open(path + '/round2B_test_img.pkl', 'rb') as handle:
        test_data = pickle.load(handle)
    with open(path + '/train_data_img_flat.pkl', 'rb') as handle:
        train_img_flat = pickle.load(handle)
    with open(path + '/test_data_img_flat.pkl', 'rb') as handle:
        test_img_flat = pickle.load(handle)
    round2_class_id = ['ZJL' + str(i) for i in range(296, 521)]
    round2_train_class_id = round2_train_img.class_id.unique()
    train_data = pd.concat([round1_train_img_part0, round1_train_img_part1, round2_train_img, 
                round2B_train_img,
                ], axis = 0, sort = False)
    train_data['target'] = list(train_img_flat)
    test_data['target'] = list(test_img_flat)
    del round1_train_img_part0, round1_train_img_part1, round2_train_img , \
                round2B_train_img \
                , train_img_flat, test_img_flat
    gc.collect()
    train_data = train_data.merge(class_id_emb_attr, how = 'left', on = 'class_id')
    if FLAGS.debug:
        train_data = pd.concat([train_data.iloc[:400], train_data.iloc[-200:]])
        test_data = test_data.iloc[-10:]

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
def ENAS(train_data):
    clf = ak.ImageClassifier(verbose = True)
    fold = FLAGS.enas_fold
    kf = KFold(n_splits=fold, shuffle=True, random_state = 100)
    for _, test_index in kf.split(train_data):
        debug_data = train_data.iloc[test_index]
        break
    print ('train size', debug_data.shape[0])
    x_train = preprocess_img(debug_data['img'])

    category = debug_data['class_id'].unique()
    print ('class size ', category.shape[0])
    category_dict = dict((category[i], i) for i in range(category.shape[0]))
    y_train = debug_data['class_id'].apply(lambda id: category_dict[id]).values
    clf.fit(x_train, y_train, time_limit = FLAGS.enas_time)

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
        print (model_path)
        model_file_name = glob.glob(model_path + '/imgmodel_*.h5')[0]
        print ('Model file name: ', model_file_name)
        img_model.load_weights(model_file_name)
        return (img_model, model_type)

    OneHotEncoder = sklearn.preprocessing.OneHotEncoder()
    train_target = train_data['class_id'].apply(lambda id: category_dict[id]).values
    # x_train = extract_array_from_series(train_data['img'])
    # y_train = train_target
    # clf = ak.ImageClassifier(verbose = True)
    # clf.fit(x_train, y_train)
    # sys.exit(0)
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
                   img_model = None,
                   fold = None,
                   ensemble_nfold = None,
                   dem_epochs = None,
                   only_emb = None,
                   model_type = None):
    print("Over all training size:")
    print(train_data.shape)

    kf = KFold(n_splits=fold, shuffle=True, random_state = 100)
    num_fold = 0
    models = []
    # model_type = FLAGS.zs_model_type
    scores = []
    classes = train_data.class_id.unique()
    if flags.load_zs_model:
        model_file_names = glob.glob(model_path + '/zsmodel_*.h5')
        for m_file in model_file_names:
            print ('Model file name: ', m_file)
            zs_model = DEM(flags = flags, model_type = model_type, 
                    img_flat_len = img_flat_len, 
                    unseen_class = classes,
                    class_id_emb_attr = class_id_emb_attr,
                    img_model = img_model,
                    dem_epochs = dem_epochs,
                    only_emb = only_emb).model
            zs_model.load_weights(m_file)
            models.append((zs_model, model_type))
        return models, None

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
                    img_model = img_model,
                    dem_epochs = dem_epochs,
                    only_emb = only_emb)
        if num_fold == 0:
            print (zs_model.model.summary())
        zs_model.train(train_part_df, validate_part_df, num_fold)
        models.append((zs_model.model, model_type))
        num_fold += 1
        if num_fold == ensemble_nfold:
            break
    score_df = pd.concat(scores, sort = False)
    # print (score_df)
    return models, score_df

# train_data['target'] = list(model_eval(img_model[0], img_model[1], train_data))
# test_data['target'] = list(model_eval(img_model[0], img_model[1], test_data))
# zs_models = train_zs_model(train_data, class_id_emb_attr, flags = FLAGS, img_flat_len = 128)

def predict_flat(img_model, train_data, test_data):
    time_label = time.strftime('%Y%m%d_%H%M%S')
    tmp_model_dir = "./model_sub/"
    if not os.path.isdir(tmp_model_dir):
        os.makedirs(tmp_model_dir, exist_ok=True)
    with open(tmp_model_dir + '/train_data_img_flat_' + time_label + '.pkl', 'wb') as handle:
        pickle.dump(extract_array_from_series(train_data['target']), handle)
    with open(tmp_model_dir + '/train_data_pred_img_class_' + time_label + '.pkl', 'wb') as handle:
        pickle.dump(extract_array_from_series(train_data['pred_img_class']), handle)
    with open(tmp_model_dir + '/test_data_img_flat_' + time_label + '.pkl', 'wb') as handle:
        pickle.dump(extract_array_from_series(test_data['target']), handle)
    with open(tmp_model_dir + '/test_data_pred_img_class_' + time_label + '.pkl', 'wb') as handle:
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
    img_flat_model = Model(inputs = img_model[0].inputs, outputs = img_model[0].get_layer(name = 'avg_pool').output)
    vote_preds, preds = multi_models_vote(models = models, eval_df = test_data, \
            cand_class_id_emb_attr = class_id_emb_attr[~class_id_emb_attr['class_id'].isin(train_id)], \
            img_feature_map = test_img_feature_map, img_model = img_flat_model, TTA = FLAGS.TTA, flags = FLAGS)
    sub = pd.DataFrame(vote_preds, index = test_data['img_id'])
    time_label = time.strftime('%Y%m%d_%H%M%S')
    tmp_model_dir = "./model_sub/"
    if not os.path.isdir(tmp_model_dir):
        os.makedirs(tmp_model_dir, exist_ok=True)
    sub_name = tmp_model_dir + "/submit_"+ time_label + ".txt"
    sub.to_csv(sub_name, header = False, sep = '\t')
    pd.DataFrame(preds, index = test_data['img_id']).to_csv(tmp_model_dir + "/preds_"+ time_label + ".txt", header = False, sep = '\t')

    if not FLAGS.load_zs_model:
        agg_dict = {}
        statistic_columns = ['mean', 'median', 'max', 'min', 'std', 'count']
        for c in score_df.columns:
            if c == 'Fold':
                continue
            agg_dict[c] = statistic_columns
        avg_score_df = score_df.groupby('Epoch').agg(agg_dict).T
        print (avg_score_df.T)
        score_df.to_csv(tmp_model_dir + '/scores.tsv')
        avg_score_df.to_csv(tmp_model_dir + '/statistic_scores.tsv')    
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

class FastTextAddArgs(FastText):
    """
    """
    @classmethod
    def train(cls, ft_path, corpus_file, output_file=None, model='cbow', size=100, alpha=0.025, window=5, min_count=5,
              word_ngrams=1, loss='ns', sample=1e-3, negative=5, iter=5, min_n=3, max_n=6, sorted_vocab=1, threads=12,
             verbose = 2, lrUpdateRate = 100):
        """
        """
        ft_path = ft_path
        output_file = output_file or os.path.join('./model_sub/')
        ft_args = {
            'input': corpus_file,
            'output': output_file,
            'lr': alpha,
            'dim': size,
            'ws': window,
            'epoch': iter,
            'minCount': min_count,
            'wordNgrams': word_ngrams,
            'neg': negative,
            'loss': loss,
            'minn': min_n,
            'maxn': max_n,
            'thread': threads,
            't': sample,
            'verbose': verbose,
            'lrUpdateRate': lrUpdateRate,
        }
        cmd = [ft_path, model]
        for option, value in ft_args.items():
            cmd.append("-%s" % option)
            cmd.append(str(value))

        gensim.utils.check_output(args=cmd)
        model = cls.load_fasttext_format(output_file)
        # cls.delete_training_files(output_file)
        return model, output_file


def train_ft():
    # Set FastText home to the path to the FastText executable
    # model = FastTextAddArgs.load(path + '/LatestCorpus_skipgram_300_10Epoch/ft')
    ft_home = './fasttext'
    os.chmod(ft_home, 0o775)
    train_file = path + '/wiki_corpus'
    tmp_model_dir = "./model_sub/"
    if not os.path.isdir(tmp_model_dir):
        os.makedirs(tmp_model_dir, exist_ok=True)
    # train the model
    model_wrapper, tmp_model_dir = FastTextAddArgs.train(ft_home, 
        train_file,
        tmp_model_dir,
        size = FLAGS.ft_size, 
        model = FLAGS.ft_model,
        threads = FLAGS.ft_threads,
        iter = FLAGS.ft_iter,
        verbose = FLAGS.ft_verbose,
        lrUpdateRate = FLAGS.ft_lrUpdateRate,
        min_count = FLAGS.ft_min_count)

    print(model_wrapper)
    print (tmp_model_dir)
    # tmp_model_dir = "./model_sub/"
    # if not os.path.isdir(tmp_model_dir):
    #     os.makedirs(tmp_model_dir, exist_ok=True)
    # model_wrapper.save(tmp_model_dir + 'ft.gz')
    if not os.path.isdir(FLAGS.output_model_path):
        os.makedirs(FLAGS.output_model_path, exist_ok=True)
    for fileName in os.listdir(tmp_model_dir):
        print (fileName)
        dst_file = os.path.join(FLAGS.output_model_path, fileName)
        if os.path.exists(dst_file):
            os.remove(dst_file)
        shutil.move(os.path.join(tmp_model_dir, fileName), FLAGS.output_model_path)

if __name__ == "__main__":
    if FLAGS.train_ft:
        train_ft()
        sys.exit(0)
    train_data, test_data, class_id_emb_attr, round2_class_id, round2_train_class_id = load_data()
    if FLAGS.enas:
        ENAS(train_data)
    else:
        img_model = train_img_classifier(train_data, flags = FLAGS)
        # if FLAGS.zs_model_type != 'DEM_AUG':
        #     train_preds = model_eval(img_model[0], img_model[1], train_data, verbose = FLAGS.train_verbose, flags = FLAGS)
        #     test_preds = model_eval(img_model[0], img_model[1], test_data, verbose = FLAGS.train_verbose, flags = FLAGS)
        #     train_data['target'] = list(train_preds[0])
        #     test_data['target'] = list(test_preds[0])
        #     train_data['pred_img_class'] = list(train_preds[1])
        #     test_data['pred_img_class'] = list(test_preds[1])
        if FLAGS.predict_flat:
            predict_flat(img_model, train_data, test_data)
        else:
            round1_class_id = list(set(train_data.class_id.unique()) - set(round2_class_id))
            if FLAGS.zs_model_type == 'RES_DEM_BC':
                zs_models, score_df = train_zs_model(train_data[train_data.class_id.isin(round1_class_id)], 
                        class_id_emb_attr = class_id_emb_attr[class_id_emb_attr.class_id.isin(round1_class_id)], 
                        flags = FLAGS, 
                        img_flat_len = FLAGS.img_flat_len,
                        round1_class_id = round1_class_id,
                        round2_class_id = round2_class_id,
                        img_model = img_model,
                        fold = FLAGS.res_dem_nfold,
                        ensemble_nfold = 1,
                        dem_epochs = FLAGS.res_dem_epochs,
                        only_emb = True,
                        model_type = 'DEM_BC')
                zs_models[0][0].save('./only_emb.h5')
            if FLAGS.only_use_round2:
                train_data = train_data[train_data.class_id.isin(round2_class_id)]
                class_id_emb_attr = class_id_emb_attr[class_id_emb_attr.class_id.isin(round2_class_id)]
            zs_models, score_df = train_zs_model(train_data, #[train_data.class_id.isin(round2_class_id)], 
                    class_id_emb_attr = class_id_emb_attr, #[class_id_emb_attr.class_id.isin(round2_class_id)], 
                    flags = FLAGS, 
                    img_flat_len = FLAGS.img_flat_len,
                    round1_class_id = round1_class_id,
                    round2_class_id = round2_class_id,
                    img_model = img_model,
                    fold = FLAGS.dem_nfold,
                    ensemble_nfold = FLAGS.dem_ensemble_nfold,
                    dem_epochs = FLAGS.dem_epochs,
                    only_emb = FLAGS.only_emb,
                    model_type = FLAGS.zs_model_type)
            cand_class_id_emb_attr = class_id_emb_attr[class_id_emb_attr.class_id.isin(round2_class_id)]
            sub(models = zs_models, train_data = train_data, test_data = test_data, 
                class_id_emb_attr = cand_class_id_emb_attr, \
                img_model = img_model, score_df = score_df)