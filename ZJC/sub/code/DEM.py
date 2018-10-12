
import sklearn
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import *
from DenseNet import DenseNet

# from tensorflow.python.keras import layers, preprocessing
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.models import Model, load_model
# from tensorflow.python.keras.callbacks import EarlyStopping, Callback
# from tensorflow.python.keras.regularizers import l1, l2
# from tensorflow.python.keras.optimizers import SGD, RMSprop, Adam, Nadam
# from tensorflow.python.keras.losses import mean_squared_error, binary_crossentropy

from keras import layers, preprocessing
from keras import backend as K
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, Callback
from keras.regularizers import l1, l2
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.losses import mean_squared_error, binary_crossentropy

class AccuracyEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1, batch_interval = 1000000, verbose = 2, \
            scores = [], cand_class_id_emb_attr = None, eval_df = None, threshold = None, \
                 seen_class = None, unseen_class = None, gamma = None, model_type = None, 
                 class_id_dict = None, class_to_id = None):
        super(AccuracyEvaluation, self).__init__()

        self.interval = interval
        # print (validation_data)
#         self.X_val, _, 
        self.y_val = validation_data[-1]
        self.verbose = verbose
        self.scores = scores
        self.cand_class_id_emb_attr = cand_class_id_emb_attr
        self.eval_df = eval_df
        self.seen_class = seen_class
        self.unseen_class = unseen_class
        self.model_type = model_type
        self.class_id_dict = class_id_dict
        self.class_to_id = class_to_id
#         self.class_id_dict['All'] = self.eval_df.class_id.unique()
        
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            epoch_scores = model_eval(self.model, self.model_type, self.eval_df, self.cand_class_id_emb_attr, 
#                 seen_class = self.seen_class, 
#                 unseen_class = self.unseen_class, 
                img_feature_map = self.y_val,
                class_id_dict = self.class_id_dict,
                class_to_id = self.class_to_id)
            self.scores.append(epoch_scores)

class DEM:
    """
    """
    def __init__(self, scores = None, flags = None, model_type = None, seen_class = None, 
            unseen_class = None, class_id_emb_attr = None, img_flat_len = None, 
                    unseen_round1_id = None,
                    unseen_round2_id = None,
                    img_model = None):
        self.batch_size = flags.dem_batch_size
        self.epochs = flags.dem_epochs
        self.patience = flags.dem_patience
        self.scores = scores
        self.model_type = model_type
        self.verbose = flags.train_verbose
        self.seen_class = seen_class
        self.unseen_class = unseen_class
        self.cand_class_id_emb_attr = class_id_emb_attr[class_id_emb_attr.class_id.isin(unseen_class)]
        self.class_id_emb_attr = class_id_emb_attr
        class_ids = class_id_emb_attr.class_id.values
        self.class_to_id = dict([(c, i) for i, c in enumerate(class_ids)])
        self.img_flat_len = img_flat_len
        if model_type == 'DEM':
            self.model = self.create_dem(img_flat_len = img_flat_len)
        elif model_type == 'GCN':
            self.model = self.create_gcn(img_flat_len = img_flat_len)
        elif model_type == 'I2A':
            self.model = self.create_img2attr(img_flat_len = img_flat_len)
        elif model_type == 'AE':
            self.model = self.create_ae(img_flat_len = img_flat_len)
        elif model_type == 'DEM_AUG':
            self.img_model = img_model
            self.rotation_range = flags.rotation_range
            self.shear_range = flags.shear_range 
            self.zoom_range = flags.zoom_range
            self.horizontal_flip = flags.horizontal_flip
            self.model = self.create_dem_aug(img_flat_len = img_flat_len)
        self.class_id_dict = {
#                              'seen_class': seen_class,
                             'Unseen_class': unseen_class,
#                              'Unseen_round1_id': unseen_round1_id,
                             'Unseen_round2_id': unseen_round2_id,}

    def create_dem(self, kernel_initializer = 'he_normal', img_flat_len = 1024):
        attr_input = layers.Input(shape = (50,), name = 'attr')
        word_emb = layers.Input(shape = (600,), name = 'wv')
        imag_classifier = layers.Input(shape = (img_flat_len,), name = 'img')
        
        attr_dense = layers.Dense(600, use_bias = True, kernel_initializer=kernel_initializer, 
                        kernel_regularizer = l2(1e-4), name = 'attr_dense')(attr_input)
        attr_word_emb = layers.Concatenate(name = 'attr_word_emb')([word_emb, attr_dense])
#         attr_word_emb_dense = self.full_connect_layer(attr_word_emb, hidden_dim = [
#                                                                             int(img_flat_len * 2),
#                                                                             int(img_flat_len * 1.5), 
#                                                                             int(img_flat_len * 1.25), 
# #                                                                             int(img_flat_len * 1.125),
# #                                                                             int(img_flat_len * 1.0625)
#                                                                             ], \
#                                                 activation = 'relu', resnet = False, drop_out_ratio = 0.2)
        attr_word_emb_dense = self.full_connect_layer(attr_word_emb_dense, hidden_dim = [img_flat_len], 
                                                activation = 'relu')

        mse_loss = K.mean(mean_squared_error(imag_classifier, attr_word_emb_dense))
        
        model = Model([attr_input, word_emb, imag_classifier], outputs = attr_word_emb_dense) #, vgg_output])
        model.add_loss(mse_loss)
        model.compile(optimizer=Adam(lr=1e-4), loss=None)
        return model
    
    def create_dem_aug(self, kernel_initializer = 'he_normal', img_flat_len = 1024):
        attr_input = layers.Input(shape = (50,), name = 'attr')
        word_emb = layers.Input(shape = (600,), name = 'wv')
        img_input = layers.Input(shape = (64, 64, 3))
#         imag_classifier = layers.Input(shape = (img_flat_len,), name = 'img')

        flat_img_model = Model(inputs = self.img_model[0].inputs, outputs = self.img_model[0].get_layer(name = 'avg_pool').output)
        flat_img_model.trainable = False
        imag_classifier = flat_img_model(img_input)
        
        attr_dense = layers.Dense(600, use_bias = True, kernel_initializer=kernel_initializer, 
                        kernel_regularizer = l2(1e-4), name = 'attr_dense')(attr_input)
        attr_word_emb = layers.Concatenate(name = 'attr_word_emb')([word_emb, attr_dense])
        attr_word_emb_dense = self.full_connect_layer(attr_word_emb, hidden_dim = [
                                                                            int(img_flat_len * 2),
                                                                            int(img_flat_len * 1.5), 
                                                                            int(img_flat_len * 1.25), 
#                                                                             int(img_flat_len * 1.125),
#                                                                             int(img_flat_len * 1.0625)
                                                                            ], \
                                                activation = 'relu', resnet = False, drop_out_ratio = 0.2)
        attr_word_emb_dense = self.full_connect_layer(attr_word_emb_dense, hidden_dim = [img_flat_len], 
                                                activation = 'relu')

        mse_loss = K.mean(mean_squared_error(imag_classifier, attr_word_emb_dense))
        
        model = Model([img_input, attr_input, word_emb], outputs = [attr_word_emb_dense, imag_classifier]) #, vgg_output])
        model.add_loss(mse_loss)
        model.compile(optimizer=Adam(lr=1e-4), loss=None)
        return model
    
    def create_img2attr(self, kernel_initializer = 'he_normal', img_flat_len = 1024):
        attr_input = layers.Input(shape = (50,), name = 'attr')
        word_emb = layers.Input(shape = (600,), name = 'wv')
        imag_classifier = layers.Input(shape = (img_flat_len,), name = 'img')

        attr_dense = layers.Dense(600, use_bias = True, kernel_initializer=kernel_initializer, 
                        kernel_regularizer = l2(1e-4), name = 'attr_dense')(attr_input)
        attr_word_emb = layers.Concatenate(name = 'attr_word_emb')([word_emb, attr_dense])
        out_size = 50
        
        attr_preds = self.full_connect_layer(imag_classifier, hidden_dim = [
                                                                            int(out_size * 20),
                                                                            int(out_size * 15), 
#                                                                             int(out_size * 7), 
#                                                                             int(img_flat_len * 1.125),
#                                                                             int(img_flat_len * 1.0625)
                                                                            ], \
                                                activation = 'relu', resnet = False, drop_out_ratio = 0.2)
        attr_preds = self.full_connect_layer(attr_preds, hidden_dim = [out_size], activation = 'sigmoid')
        log_loss = K.mean(binary_crossentropy(attr_input, attr_preds))
        
        model = Model([attr_input, word_emb, imag_classifier], outputs = [attr_preds]) #, vgg_output])
        model.add_loss(log_loss)
        model.compile(optimizer=Adam(lr=1e-5), loss=None)
        return model
    
    def create_ae(self, kernel_initializer = 'he_normal', img_flat_len = 1024):
        gamma = 0.5
        attr_input = layers.Input(shape = (50,), name = 'attr')
        word_emb = layers.Input(shape = (600,), name = 'wv')
        imag_classifier = layers.Input(shape = (img_flat_len,), name = 'img')

        attr_dense = layers.Dense(600, use_bias = True, kernel_initializer=kernel_initializer, 
                        kernel_regularizer = l2(1e-4), name = 'attr_dense')(attr_input)
#         attr_dense = self.full_connect_layer(attr_dense, hidden_dim = [int(img_flat_len * 1.5), 
#                                                                             int(img_flat_len * 1.25), 
# #                                                                             int(img_flat_len * 1.125),
# #                                                                               int(img_flat_len * 0.5)
#                                                                             ], \
#                                                 activation = 'relu', resnet = False, drop_out_ratio = 0.2)
        attr_word_emb = layers.Concatenate(name = 'attr_word_emb')([word_emb, attr_dense])
        attr_word_emb_dense = self.full_connect_layer(attr_word_emb, hidden_dim = [
                                                                            int(img_flat_len * 2),
                                                                            int(img_flat_len * 1.5), 
                                                                            int(img_flat_len * 1.25), 
#                                                                             int(img_flat_len * 1.125),
#                                                                             int(img_flat_len * 1.0625)
                                                                            ], \
                                                activation = 'relu', resnet = False, drop_out_ratio = 0.2)
        attr_word_emb_dense = self.full_connect_layer(attr_word_emb_dense, hidden_dim = [img_flat_len], 
                                                activation = 'relu')

        mse_loss = K.mean(mean_squared_error(imag_classifier, attr_word_emb_dense))
        
        out_size = 50
        attr_preds = self.full_connect_layer(attr_word_emb_dense, hidden_dim = [
                                                                            int(out_size * 20),
                                                                            int(out_size * 15), 
                                                                            int(out_size * 7), 
#                                                                             int(img_flat_len * 1.125),
#                                                                             int(img_flat_len * 1.0625)
                                                                            ], \
                                                activation = 'relu', resnet = False, drop_out_ratio = 0.2)
        attr_preds = self.full_connect_layer(attr_preds, hidden_dim = [out_size], activation = 'sigmoid')
        log_loss = K.mean(binary_crossentropy(attr_input, attr_preds))
        
        loss = (1 - gamma) * mse_loss + gamma * log_loss
        
        model = Model([attr_input, word_emb, imag_classifier], outputs = [attr_word_emb_dense, attr_preds])
        model.add_loss(loss)
        model.compile(optimizer=Adam(lr=1e-4), loss=None)
        return model
    
    def create_gcn(self, img_flat_len = 1024):
        adj_graph = 1 - sklearn.metrics.pairwise.pairwise_distances(
            np.array(list(self.class_id_emb_attr['emb']))[:, :300], metric = 'cosine')
        attr_input = layers.Input(tensor=
                            tf.constant(np.array(list(self.class_id_emb_attr['attr']), 
                                                 dtype = 'float32')))
        all_word_emb = layers.Input(tensor=
                        tf.constant(extract_array_from_series(self.class_id_emb_attr['emb']), 
                                    dtype = 'float32')) #Input(shape = (230, 300,), name = 'wv')
        class_index = layers.Input(shape = (1, ), name = 'class_index', dtype = 'int32')
        adj_graphs = layers.Input(tensor=tf.constant(adj_graph, dtype = 'float32')) #Input(shape = (230, 230,), name = 'adj_graph')
        imag_classifier = layers.Input(shape = (img_flat_len,), name = 'img')

        attr_dense = layers.Dense(600, use_bias = False, kernel_initializer='he_normal', 
                        kernel_regularizer = l2(1e-4))(attr_input)
        attr_word_emb = layers.Concatenate()([all_word_emb, attr_dense])
        
        all_classifier = self.full_connect_layer(attr_word_emb, hidden_dim = [
                                                                        int(img_flat_len * 2),
                                                                        int(img_flat_len * 1.5), 
                                                                        int(img_flat_len * 1.25 ),
#                                                                         img_flat_len
                                                                            ], 
                                    activation = 'relu', adj_graphs = adj_graphs, drop_out_ratio = 0.2)
        all_classifier = self.full_connect_layer(all_classifier, hidden_dim = [img_flat_len], 
                                    activation = 'relu', adj_graphs = adj_graphs)
        x = tf.gather_nd(all_classifier, class_index)

        mse_loss = K.mean(mean_squared_error(imag_classifier, x))

        model = Model([class_index, imag_classifier, attr_input, all_word_emb, adj_graphs], 
                      outputs = [all_classifier]) #, vgg_output])
        model.add_loss(mse_loss)
        model.compile(optimizer=Adam(lr=1e-4), loss=None)
    #     model.summary()
        return model

    def full_connect_layer(self, input, hidden_dim, activation, resnet = False, adj_graphs = None, 
                        drop_out_ratio = None, kernel_initializer = 'he_normal'):
        full_connect = input
        for i, hn in enumerate(hidden_dim):
            fc_in = full_connect
            if drop_out_ratio is not None:
                full_connect = layers.Dropout(drop_out_ratio)(full_connect)
            full_connect = layers.BatchNormalization(epsilon=1.001e-5)(full_connect)
            full_connect = layers.Dense(hn, kernel_initializer=kernel_initializer, kernel_regularizer = l2(1e-4), 
                    activation = None)(full_connect)
            if adj_graphs is not None:
                full_connect = layers.Lambda(lambda x: K.dot(x[1], x[0]))([full_connect, adj_graphs])
            full_connect = layers.Activation(activation)(full_connect)
            if resnet:
                full_connect = layers.Concatenate()([fc_in, full_connect])
        return full_connect

    def DNN_DataSet(self, df):
        """
        """
        if self.model_type == 'DEM' or self.model_type == 'I2A' or self.model_type == 'AE':
            return create_dem_data(df) + [extract_array_from_series(df['target'])]
        elif self.model_type == 'DEM_AUG':
            return [preprocess_img(df['img'])] + create_dem_data(df)
        elif self.model_type == 'GCN':
            return [create_gcn_data(df, self.class_to_id), extract_array_from_series(df['target'])]

    def train(self, train_part_df, validate_part_df):
        """
        Keras Training
        """
        print("-----DNN training-----")

        DNN_Train_Data = self.DNN_DataSet(train_part_df)
        DNN_validate_Data = self.DNN_DataSet(validate_part_df)
        scores_list = []
        callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=0),
        AccuracyEvaluation(validation_data=DNN_validate_Data, interval=1,
                            cand_class_id_emb_attr = self.cand_class_id_emb_attr,
                            eval_df = validate_part_df,
                            model_type = self.model_type,
                            class_id_dict = self.class_id_dict,
                            class_to_id = self.class_to_id,
                            scores = scores_list)
        ]
        if self.model_type == 'DEM_AUG':
            datagen = MixedImageDataGenerator(
                    rotation_range=self.rotation_range,
                    shear_range = self.shear_range,
                    zoom_range=self.zoom_range,
                    horizontal_flip=self.horizontal_flip)
            datagen.fit(DNN_Train_Data[0])
            h = self.model.fit_generator(
                    datagen.flow((DNN_Train_Data[0], DNN_Train_Data[1:]), None, batch_size=self.batch_size), 
                    validation_data=(DNN_validate_Data, None), steps_per_epoch = DNN_Train_Data[0].shape[0]//self.batch_size,
                    epochs=self.epochs, shuffle=True, verbose = self.verbose, workers=1, use_multiprocessing=False, 
                    callbacks=callbacks)
        else:
            h = self.model.fit(DNN_Train_Data,  validation_data = (DNN_validate_Data, None),
                        epochs=self.epochs, batch_size = self.batch_size, shuffle=True, verbose = self.verbose, callbacks=callbacks)
        self.scores.append(pd.DataFrame(scores_list[-1:], columns = self.class_id_dict.keys))
        return self.model

    def predict(self, test_part, batch_size = 1024, verbose=2):
        """
        Keras Training
        """
        print("-----DNN Test-----")
        pred = self.model.predict(self.DNN_DataSet(test_part), verbose=verbose)
        if self.model_type == 'r':
            pred = pred[:, -1]
        return pred