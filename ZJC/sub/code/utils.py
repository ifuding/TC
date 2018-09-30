import numpy as np
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.models import Model

def extract_array_from_series(s):
    return np.asarray(list(s))

def create_dem_data(df):
    # return [extract_array_from_series(df['attr'])[:, :30], extract_array_from_series(df['emb'])[:, :]]
    return [extract_array_from_series(df['attr'])[:, :], extract_array_from_series(df['emb'])[:, :]]

def preprocess_img(img_series):
    return vgg16.preprocess_input(extract_array_from_series(img_series))

def find_nearest_class(class_id_emb_attr, eval_df, cand_feature_map, img_feature_map):
    nearest_class_id = ['ZJL'] * eval_df.shape[0]
    for i in range(img_feature_map.shape[0]):
        img = img_feature_map[i]
        dis = np.linalg.norm(img - cand_feature_map, axis = 1)
        min_ind = np.where(dis == np.amin(dis))[0]
        nearest_class_id[i] = class_id_emb_attr.iloc[min_ind[0]]['class_id']
    return np.asarray(nearest_class_id)
        
def calc_accuracy(eval_df, eval_class, preds):
    eval_mask = eval_df.class_id.isin(eval_class)
    eval_num = np.sum(eval_mask)
    right_num = np.sum(preds[eval_mask] == eval_df.class_id[eval_mask])
    return right_num / np.sum(eval_mask), right_num, eval_num
    
def calc_detailed_accuracy(eval_df, preds, seen_class, unseen_class):
    all_re = calc_accuracy(eval_df, eval_df['class_id'].values, preds)
    seen_re = calc_accuracy(eval_df, seen_class, preds)
    unseen_re = calc_accuracy(eval_df, unseen_class, preds)
    print("\nAll_re: \t%.6f\t%.0f\t%.0f" % all_re)
    print("Seen_re: \t%.6f\t%.0f\t%.0f" % seen_re)
    print("Unseen_re: \t%.6f\t%.0f\t%.0f" % unseen_re)
        
def multi_models_vote(models, eval_df = None, cand_class_id_emb_attr = None, img_feature_map = None, 
                      seen_class = None, unseen_class = None):
    print ('cand shape: ', cand_class_id_emb_attr.shape[0])
    preds = models_eval(models, eval_df, cand_class_id_emb_attr, img_feature_map, 
                seen_class, unseen_class)
    preds = np.asarray(preds).T
    # print (preds)
    vote_preds = []
    for single_img_vote in preds:
        uniq_val, counts = np.unique(single_img_vote, return_counts = True)
        vote_preds.append(uniq_val[np.argmax(counts)])
    vote_preds = np.asarray(vote_preds)
    # print (vote_preds)
    if 'class_id' in eval_df.columns: 
        calc_detailed_accuracy(eval_df, vote_preds, seen_class, unseen_class)
    return vote_preds

def model_eval(model, model_type, eval_df, cand_class_id_emb_attr = None, img_feature_map = None, 
                seen_class = None, unseen_class = None):
    """
    """
    if model_type == 'DenseNet':
        flat_model = Model(inputs = model.inputs, outputs = model.get_layer(name = 'avg_pool').output)
        pred = flat_model.predict(preprocess_img(eval_df['img']))
    elif model_type == 'DEM':
        zs_model = Model(inputs = model.inputs[:2], outputs = model.outputs[0])
        cand_feature_map = zs_model.predict(create_dem_data(cand_class_id_emb_attr))
        pred = find_nearest_class(cand_class_id_emb_attr, eval_df, cand_feature_map, img_feature_map)
        if 'class_id' in eval_df.columns:
            calc_detailed_accuracy(eval_df, pred, seen_class, unseen_class)
    return pred

def models_eval(models, eval_df, cand_class_id_emb_attr = None, img_feature_map = None, 
                seen_class = None, unseen_class = None):
    preds = []
    for (model, model_type) in models:
        pred = model_eval(model, model_type, eval_df = eval_df, cand_class_id_emb_attr = cand_class_id_emb_attr, 
            img_feature_map = img_feature_map, seen_class = seen_class, unseen_class = unseen_class)
        preds.append(pred)
    return preds