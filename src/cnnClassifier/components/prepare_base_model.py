import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape= self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path = self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model,classes,freeze_all,freeze_till,learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till >0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)

        dropout = tf.keras.layers.Dropout(0.5)(flatten_in)

        dense = tf.keras.layers.Dense(256, activation='relu')(dropout)
        dense = tf.keras.layers.Dropout(0.3)(dense)


        prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(dense)

        full_model = tf.keras.models.Model(
            model.input,  
            prediction    
        )
        
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_k_categorical_accuracy')
        ]


        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=metrics
        )

        full_model.summary()
        return full_model
    
    def updated_base_model(self):
        self.full_model =self._prepare_full_model(
            model= self.model,
            classes = self.config.params_classes,
            freeze_all = False,
            freeze_till = 4,
            learning_rate = self.config.params_learning_rate

        )

        self.save_model(path = self.config.updated_base_model_path, model=self.full_model)


