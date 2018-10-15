import numpy as np
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from tensorflow.python.keras import backend as K

# from keras.applications import vgg16 #, densenet
# from keras.models import Model
# from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
# from keras import backend as K

def extract_array_from_series(s):
    return np.asarray(list(s))

def create_dem_data(df):
    # return [extract_array_from_series(df['attr'])[:, :30], extract_array_from_series(df['emb'])[:, :]]
    return [extract_array_from_series(df['attr'])[:, :50], extract_array_from_series(df['emb'])[:, :]]

def create_gcn_data(df, class_to_id):
    return np.array([class_to_id[c] for c in df['class_id'].values]).astype('int32')

def preprocess_numpy_input(x, data_format = 'channels_last', mode = 'torch', **kwargs):
    """Preprocesses a Numpy array encoding a batch of images.

    # Arguments
        x: Input array, 3D or 4D.
        data_format: Data format of the image array.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.

    # Returns
        Preprocessed Numpy array.
    """
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(K.floatx(), copy=False)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
        return x

    if mode == 'torch':
        x /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            if x.ndim == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x

def preprocess_img(img_series):
    return preprocess_numpy_input(extract_array_from_series(img_series))
    # return densenet.preprocess_input(extract_array_from_series(img_series))

def multi_labels_cross_entropy(y_true, y_preds, eps = 1e-6):
#     multi_labels_loss = [sklearn.metrics.log_loss]
#     y_preds_clip = max(eps, min(1 - eps, y_preds))
    y_preds_clip = np.clip(y_preds, eps, 1- eps)
    multi_loss = -(y_true * np.log(y_preds_clip) + (1 - y_true) * np.log(1 - y_preds_clip))
    return np.mean(multi_loss, axis = -1)

def find_nearest_class(class_id_emb_attr, eval_df, cand_feature_map, img_feature_map,
                      model_type = None, attr_preds = None):
    nearest_class_id = ['ZJL'] * img_feature_map.shape[0]
    for i in range(img_feature_map.shape[0]):
        if model_type == 'I2A':
            dis = multi_labels_cross_entropy(extract_array_from_series(class_id_emb_attr['attr'])[:, :50],
                                            attr_preds[i])
        else:
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
    
def calc_detailed_accuracy(eval_df, preds, class_id_dict):
    print ("\n")
    scores = []
    for class_set_name in sorted(class_id_dict):
        class_set = class_id_dict[class_set_name]
        re = calc_accuracy(eval_df, class_set, preds)
        print("%s: \t%.6f\t%.0f\t%.0f" % ((class_set_name,) + re))
        scores.append(re[0])
    return scores
    # print ("\n")
        
def multi_preds_vote(preds):
    vote_preds = []
    for single_img_vote in preds:
        uniq_val, counts = np.unique(single_img_vote, return_counts = True)
        vote_preds.append(uniq_val[np.argmax(counts)])
    vote_preds = np.asarray(vote_preds)
    return vote_preds
    
def multi_models_vote(models, eval_df = None, cand_class_id_emb_attr = None, img_feature_map = None, 
                      class_id_dict = None, img_model = None, TTA = None, flags = None):
    print ('cand shape: ', cand_class_id_emb_attr.shape[0])
    preds = models_eval(models, eval_df, cand_class_id_emb_attr, img_feature_map, 
                class_id_dict, img_model = img_model, TTA = TTA, flags = flags)
    preds = np.asarray(preds).T
    vote_preds = multi_preds_vote(preds)
    # print (vote_preds)
    if 'class_id' in eval_df.columns: 
        calc_detailed_accuracy(eval_df, vote_preds, class_id_dict)
    return vote_preds

def model_eval(model, model_type, eval_df, cand_class_id_emb_attr = None, img_feature_map = None, 
                class_id_dict = None, class_to_id = None, verbose = 2, img_model = None, TTA = None,
                flags = None):
    """
    """
    if model_type == 'DenseNet':
        flat_model = Model(inputs = model.inputs, outputs = model.get_layer(name = 'avg_pool').output)
        pred_flat = flat_model.predict(preprocess_img(eval_df['img']), verbose = verbose)
        pred_proba = model.predict(preprocess_img(eval_df['img']), verbose = verbose)
        pred = (pred_flat, pred_proba)
    else:
        if model_type == 'DEM' or model_type == 'AE':
            zs_model = Model(inputs = model.inputs[:2], outputs = model.outputs[0])
            cand_feature_map = zs_model.predict(create_dem_data(cand_class_id_emb_attr), verbose = verbose)
            if TTA is not None:
                batch_size = 128
                datagen = MixedImageDataGenerator(
                        rotation_range = flags.rotation_range,
                        shear_range = flags.shear_range,
                        zoom_range = flags.zoom_range,
                        horizontal_flip = flags.horizontal_flip)
                img_feature_map = img_model.predict_generator(
                    datagen.flow((preprocess_img(eval_df['img'])), None, shuffle = False, batch_size = batch_size), 
                    steps = np.ceil(eval_df.shape[0] / batch_size) * TTA, verbose = verbose)
                pred = find_nearest_class(cand_class_id_emb_attr, eval_df, cand_feature_map, img_feature_map)
                pred = np.reshape(pred, (TTA, eval_df.shape[0])).T
                pred = multi_preds_vote(pred)
            else:
                pred = find_nearest_class(cand_class_id_emb_attr, eval_df, cand_feature_map, img_feature_map)
        elif model_type == 'GCN':
            zs_model = Model(inputs = model.inputs[2:], outputs = model.outputs[0])
            cand_class_to_id = [class_to_id[c] for c in cand_class_id_emb_attr.class_id.values]
            cand_feature_map = zs_model.predict(None, steps = 1)[cand_class_to_id]
            pred = find_nearest_class(cand_class_id_emb_attr, eval_df, cand_feature_map, img_feature_map)
        elif model_type == 'I2A':
            zs_model = Model(inputs = model.inputs[-1], outputs = model.outputs[0])
            attr_preds = zs_model.predict(extract_array_from_series(img_feature_map), verbose = verbose)
            pred = find_nearest_class(cand_class_id_emb_attr, eval_df, None, img_feature_map, 
                                    model_type, attr_preds)
        elif model_type == 'DEM_AUG':
            img_model = Model(inputs = model.inputs[0], outputs = model.outputs[-1])
            img_feature_map = img_model.predict(preprocess_img(eval_df['img']), verbose = verbose)
            zs_model = Model(inputs = model.inputs[1:], outputs = model.outputs[0])
            cand_feature_map = zs_model.predict(create_dem_data(cand_class_id_emb_attr), verbose = verbose)
            pred = find_nearest_class(cand_class_id_emb_attr, eval_df, cand_feature_map, img_feature_map)
        scores = None
        if 'class_id' in eval_df.columns:
            scores = calc_detailed_accuracy(eval_df, pred, class_id_dict)
        pred = (pred, scores)
    return pred

def models_eval(models, eval_df, cand_class_id_emb_attr = None, img_feature_map = None, 
                class_id_dict = None, class_to_id = None, img_model = None, TTA = None,
                flags = None):
    preds = []
    for (model, model_type) in models:
        pred, _ = model_eval(model, model_type, eval_df = eval_df, cand_class_id_emb_attr = cand_class_id_emb_attr, 
            img_feature_map = img_feature_map, class_id_dict = class_id_dict, class_to_id = class_to_id,
            img_model = img_model, TTA = TTA, flags = flags)
        preds.append(pred)
    return preds

class MixedImageDataGenerator(ImageDataGenerator):
    def __init__(self, **kwargs):
        super(MixedImageDataGenerator, self).__init__(**kwargs)
        
    def flow(self,
           x,
           y=None,
           batch_size=32,
           shuffle=True,
           seed=None,
           save_to_dir=None,
           save_prefix='',
           save_format='png'):
        return MixedNumpyArrayIterator(
        x,
        y,
        self,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        data_format=self.data_format,
        save_to_dir=save_to_dir,
        save_prefix=save_prefix,
        save_format=save_format)


class MixedNumpyArrayIterator(NumpyArrayIterator):
    """Iterator yielding data from a Numpy array.
    """

    def __init__(self,
               x,
               y,
               image_data_generator,
               **kwargs):
        self.x_misc = None
        if (type(x) is list) or (type(x) is tuple):
            super(MixedNumpyArrayIterator, self).__init__(x[0],
                   y,
                   image_data_generator,
                   **kwargs)
            self.x_misc = [np.asarray(xx) for xx in x[1]]
        else:
            super(MixedNumpyArrayIterator, self).__init__(x,
               y,
               image_data_generator,
               **kwargs)
            

    def next(self):
        """For python 2.x.

        Returns:
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(
              self.index_generator)
#         print (index_array)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(
            tuple([current_batch_size] + list(self.x.shape)[1:]), dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
#             x_bak = x
            x = self.image_data_generator.random_transform(x.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)
#             print (np.all(x_bak == x))
            batch_x[i] = x
        if self.x_misc is None:
            batch_x = [batch_x]
        else:
            batch_x_miscs = [xx[index_array] for xx in self.x_misc]
            batch_x = [batch_x] + batch_x_miscs
        return (batch_x, None, None)