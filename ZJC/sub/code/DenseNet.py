
from sklearn import metrics, preprocessing, pipeline, \
    feature_extraction, decomposition, model_selection
import sklearn
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import preprocess_img

from tensorflow.python.keras import layers, preprocessing
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.callbacks import EarlyStopping, Callback
from tensorflow.python.keras.regularizers import l1, l2
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adam, Nadam

# from keras import layers, preprocessing
# from keras import backend as K
# from keras.models import Model, load_model
# from keras.callbacks import EarlyStopping, Callback
# from keras.regularizers import l1, l2
# from keras.optimizers import SGD, RMSprop, Adam, Nadam
K.set_image_data_format('channels_last')

class DenseNet:
    """
    """
    def __init__(self, scores = None, cat_max = None, flags = None, model_type = None):
        self.batch_size = flags.densenet_batch_size
        self.epochs = flags.densenet_epochs
        self.patience = flags.densenet_patience
        self.scores = scores
        self.cat_max = cat_max
        self.model_type = model_type
        self.aug_data = flags.aug_data
        self.lr = flags.lr
        self.verbose = flags.train_verbose
        self.OneHotEncoder = sklearn.preprocessing.OneHotEncoder()
        self.rotation_range = flags.rotation_range
        self.shear_range = flags.shear_range 
        self.zoom_range = flags.zoom_range
        self.horizontal_flip = flags.horizontal_flip
        self.model = self.small_densenet(
                blocks = [int(b.strip()) for b in flags.blocks.strip().split(',')], 
                weight_decay = flags.weight_decay, 
                kernel_initializer = flags.kernel_initializer,
                init_filters = flags.init_filters,
                reduction = flags.reduction,
                growth_rate = flags.growth_rate,
                init_stride = flags.init_stride,
                img_input_shape = (flags.pixel, flags.pixel, 3))

    def dense_block(self, x, blocks, name, 
            weight_decay = 1e-4, 
            kernel_initializer = 'he_normal',
            growth_rate = None):
        """A dense block.
        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.
        # Returns
            output tensor for the block.
        """
        for i in range(blocks):
            x = self.conv_block(x, growth_rate, name=name + '_block' + str(i + 1), 
                weight_decay = weight_decay,
                kernel_initializer = kernel_initializer)
        return x

    def transition_block(self, x, reduction, name, weight_decay = 1e-4, kernel_initializer = 'he_normal'):
        """A transition block.
        # Arguments
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.
        # Returns
            output tensor for the block.
        """
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                    name=name + '_bn')(x)
        x = layers.Activation('relu', name=name + '_relu')(x)
        x = layers.Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1,
                        use_bias=False,
                        kernel_initializer = kernel_initializer,
                        kernel_regularizer = l2(weight_decay),
                        name=name + '_conv')(x)
        x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        return x

    def conv_block(self, x, growth_rate, name, weight_decay = 1e-4, kernel_initializer = 'he_normal'):
        """A building block for a dense block.
        # Arguments
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.
        # Returns
            Output tensor for the block.
        """
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        x1 = layers.BatchNormalization(axis=bn_axis,
                                    epsilon=1.001e-5,
                                    name=name + '_0_bn')(x)
        x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
        x1 = layers.Conv2D(4 * growth_rate, 1,
                        use_bias=False,
                        kernel_initializer = kernel_initializer,
                        kernel_regularizer = l2(weight_decay),
                        name=name + '_1_conv')(x1)
        x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                    name=name + '_1_bn')(x1)
        x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
        x1 = layers.Conv2D(growth_rate, 3,
                        padding='same',
                        use_bias=False,
                        kernel_initializer = kernel_initializer,
                        kernel_regularizer = l2(weight_decay),
                        name=name + '_2_conv')(x1)
        x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
        return x

    def small_densenet(self, img_input_shape = (64, 64, 3), 
        blocks = [6, 12, 24, 16], 
        weight_decay = 1e-4, 
        kernel_initializer = 'he_normal',
        init_filters = None,
        reduction = None,
        growth_rate = None,
        init_stride = None
        ):
        img_input = layers.Input(shape = (img_input_shape))

        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
        x = layers.Conv2D(init_filters, 3, strides=init_stride, use_bias=False, 
            kernel_initializer = kernel_initializer, 
            kernel_regularizer = l2(weight_decay),
            name='conv1/conv')(x)
        x = layers.BatchNormalization(
            axis=3, epsilon=1.001e-5, name='conv1/bn')(x)
        x = layers.Activation('relu', name='conv1/relu')(x)
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = layers.AveragePooling2D(3, strides=2, name='pool1')(x)
        
        for i, block in enumerate(blocks):
            scope_num_str = str(i + 2)
            x = self.dense_block(x, block, name='conv' + scope_num_str, 
                                 growth_rate = growth_rate,
                                 weight_decay = weight_decay, 
                                 kernel_initializer = kernel_initializer)
            if i != len(blocks) - 1:
                x = self.transition_block(x, reduction, name='pool' + scope_num_str, 
                                          weight_decay = weight_decay, kernel_initializer = kernel_initializer)
        x = layers.BatchNormalization(
            axis=3, epsilon=1.001e-5, name='bn')(x)
        x = layers.Activation('relu', name='relu')(x)

        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(self.cat_max, activation='softmax',
            kernel_initializer = kernel_initializer, 
            name='fc')(x)
        
        model = Model(img_input, x)
        model.compile(optimizer = Adam(lr=self.lr), loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
        
        return model

    def DNN_DataSet(self, df):
        """
        """
        return preprocess_img(df['img'])

    def train(self, train_part_df, train_part_label, validate_part_df, validate_part_label):
        """
        Keras Training
        """
        print("-----DNN training-----")

        DNN_Train_Data = self.DNN_DataSet(train_part_df)
        DNN_validate_Data = self.DNN_DataSet(validate_part_df)

        callbacks = [
                EarlyStopping(monitor='val_categorical_accuracy', patience=self.patience, verbose=0),
                ]
        if self.aug_data:
            datagen = preprocessing.image.ImageDataGenerator(
                    rotation_range=self.rotation_range,
                    shear_range = self.shear_range,
                    zoom_range=self.zoom_range,
                    horizontal_flip=self.horizontal_flip)

            datagen.fit(DNN_Train_Data)

            h = self.model.fit_generator(datagen.flow(DNN_Train_Data, train_part_label, batch_size=self.batch_size), 
                    validation_data=(DNN_validate_Data, validate_part_label), steps_per_epoch = DNN_Train_Data.shape[0]//self.batch_size,
                    epochs=self.epochs, shuffle=True, verbose = self.verbose, workers=1, use_multiprocessing=False, 
                    callbacks=callbacks)
        else:
            h = self.model.fit(DNN_Train_Data, train_part_label, batch_size=self.batch_size, epochs=self.epochs,
                        shuffle=True, verbose=self.verbose,
                        validation_data=(DNN_validate_Data, validate_part_label)
                        , callbacks=callbacks
                        )
        self.scores.append(pd.DataFrame(h.history))
        return self.model

    def predict(self, test_part, batch_size = 1024, verbose=2):
        """
        Keras Training
        """
        print("-----DNN Test-----")
        pred = self.model.predict(self.DNN_DataSet(test_part), verbose=verbose)
        return pred