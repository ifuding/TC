
import sklearn
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import *

from tensorflow.python.keras import layers, preprocessing
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.callbacks import EarlyStopping, Callback
from tensorflow.python.keras.regularizers import l1, l2
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adam, Nadam
from tensorflow.python.keras.losses import mean_squared_error, binary_crossentropy
    
class AccuracyEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1, batch_interval = 1000000, verbose = 2, \
            scores = [], class_id_emb_attr = None, eval_df = None, threshold = None, \
                 seen_class = None, unseen_class = None, gamma = None, model_type = None):
        super(Callback, self).__init__()

        self.interval = interval
        # print (validation_data)
        self.X_val, _, self.y_val = validation_data
        self.verbose = verbose
        self.scores = scores
        self.class_id_emb_attr = class_id_emb_attr
        self.eval_df = eval_df
        self.seen_class = seen_class
        self.unseen_class = unseen_class
        self.model_type = model_type
        
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            model_eval(self.model, self.model_type, self.eval_df, self.class_id_emb_attr, 
                seen_class = self.seen_class, unseen_class = self.unseen_class, img_feature_map = self.y_val)

class DEM:
    """
    """
    def __init__(self, scores = None, flags = None, model_type = None, seen_class = None, 
            unseen_class = None, class_id_emb_attr = None, img_flat_len = None):
        self.batch_size = flags.dem_batch_size
        self.epochs = flags.dem_epochs
        self.patience = flags.dem_patience
        self.scores = scores
        self.model_type = model_type
        self.verbose = flags.train_verbose
        self.seen_class = seen_class
        self.unseen_class = unseen_class
        self.class_id_emb_attr = class_id_emb_attr
        self.model = self.create_dem(img_flat_len = img_flat_len)

    def create_dem(self, kernel_initializer = 'he_normal', img_flat_len = 1024):
        attr_input = layers.Input(shape = (50,), name = 'attr')
        word_emb = layers.Input(shape = (600,), name = 'wv')
        imag_classifier = layers.Input(shape = (img_flat_len,), name = 'img')

        attr_dense = layers.Dense(600, use_bias = False, kernel_initializer=kernel_initializer, 
                        kernel_regularizer = l2(1e-4), name = 'attr_dense')(attr_input)
        attr_word_emb = layers.Concatenate(name = 'attr_word_emb')([word_emb, attr_dense])
        attr_word_emb_dense = self.full_connect_layer(attr_word_emb, hidden_dim = [
                                                                            int(img_flat_len * 2),
                                                                            int(img_flat_len * 1.5), 
                                                                            int(img_flat_len * 1.25), 
                                                                            # int(img_flat_len * 1.125),
    #                                                                           int(img_flat_len * 0.5)
                                                                            ], \
                                                activation = 'relu', resnet = False, drop_out_ratio = 0.2)
        attr_word_emb_dense = self.full_connect_layer(attr_word_emb_dense, hidden_dim = [img_flat_len], 
                                                activation = 'relu')

        mse_loss = K.mean(mean_squared_error(imag_classifier, attr_word_emb_dense))
        
        model = Model([attr_input, word_emb, imag_classifier], outputs = attr_word_emb_dense) #, vgg_output])
        model.add_loss(mse_loss)
        model.compile(optimizer=Adam(lr=1e-4), loss=None)
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
                    activation = activation)(full_connect)
            if adj_graphs is not None:
                full_connect = layers.Lambda(lambda x: K.dot(x[1], x[0]), \
                                    name = 'rela_' + str(i))([full_connect, adj_graphs])
            if resnet:
                full_connect = layers.Concatenate()([fc_in, full_connect])
        return full_connect

    def DNN_DataSet(self, df):
        """
        """
        return create_dem_data(df) + [extract_array_from_series(df['target'])]

    def train(self, train_part_df, validate_part_df):
        """
        Keras Training
        """
        print("-----DNN training-----")

        DNN_Train_Data = self.DNN_DataSet(train_part_df)
        DNN_validate_Data = self.DNN_DataSet(validate_part_df)
        
        callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=0),
        AccuracyEvaluation(validation_data=DNN_validate_Data, interval=1,
                            class_id_emb_attr = self.class_id_emb_attr,
                            eval_df = validate_part_df,
                            seen_class = self.seen_class, unseen_class = self.unseen_class,
                            model_type = self.model_type)
        ]
        h = self.model.fit(DNN_Train_Data,  validation_data = (DNN_validate_Data, None),
                    epochs=self.epochs, batch_size = self.batch_size, shuffle=True, verbose = self.verbose, callbacks=callbacks)
        self.scores.append(pd.DataFrame(h.history))
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