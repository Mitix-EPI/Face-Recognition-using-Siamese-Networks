import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionResNetV2

from tensorflow.keras import layers, models
from keras.regularizers import l2

tf.random.set_seed(42)

class SiameseNetwork():

    def __init__(self,
                 IMG_SHAPE,
                 cnn_filters=[64,64,128],
                 epochs=15,
                 lr=1e-4,
                 batch_size=32,
                 dense_neurons=100,
                 activation_fct="relu",
                 fine_tunning=False,
                 dropout_rate=0.3) -> None:
        self.IMG_SHAPE = IMG_SHAPE
        self.cnn_filters = cnn_filters
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dense_neurons = dense_neurons
        self.activation_fct = activation_fct
        self.fine_tunning = fine_tunning
        self.dropout_rate = dropout_rate
        self.model = None

    def get_siamese_network(self):
        input_a = layers.Input(shape=self.IMG_SHAPE)
        input_b = layers.Input(shape=self.IMG_SHAPE)

        if self.fine_tunning is False:
            prediction_model = self._get_siamese_network_from_scratch(input_a, input_b)
        else:
            prediction_model = self._get_fine_tunning_siamese_network(input_a, input_b)

        self.model = models.Model([input_a, input_b], prediction_model)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss="binary_crossentropy", metrics=["accuracy"])
        return self.model

    def train(self, X_train, y_train, X_val, y_val):
        history = self.model.fit(X_train, y_train,
            validation_data=(X_val, y_val), batch_size=self.batch_size, epochs=self.epochs)
        return history
    
    def evaluate(self, X_test, y_test):
        test_loss, test_acc = self.model.evaluate(X_test, y_test, batch_size=self.batch_size)
        return test_loss, test_acc

    def predict(self, X):
        preds = self.model.predict(X)
        return preds

    def _get_CNN(self):
        model = models.Sequential(name="CNN")

        for i in range(len(self.cnn_filters)):
            if i == 0:
                model.add(layers.Conv2D(self.cnn_filters[i], (3, 3), input_shape=self.IMG_SHAPE, activation='relu', padding='same', kernel_regularizer=l2(1e-2)))
            else:
                model.add(layers.Conv2D(self.cnn_filters[i], (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-2)))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D())
            model.add(layers.Dropout(self.dropout_rate))

        model.add(layers.Flatten())
        model.add(layers.Dense(self.dense_neurons, activation="sigmoid", kernel_regularizer=l2(1e-4))) # * Sigmoid at start

        return model

    def _get_siamese_network_from_scratch(self, input_a, input_b):
        model = self._get_CNN()

        processed_a = model(input_a)
        processed_b = model(input_b)

        # Compute distance between the two output layers
        distance = layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([processed_a, processed_b])
        dist_dropout = layers.Dropout(self.dropout_rate)(distance)
        prediction = layers.Dense(1, activation='sigmoid')(dist_dropout)
        return prediction

    def _get_fine_tunning_siamese_network(self, input_a, input_b):
        # Load ResNet50 with imagenet weights
        resnet = ResNet50(input_shape=self.IMG_SHAPE, weights='imagenet', include_top=False)

        for layer in resnet.layers:
            layer.trainable = False

        processed_a = resnet(input_a)
        processed_b = resnet(input_b)

        flatten_a = layers.Flatten()(processed_a)
        flatten_b = layers.Flatten()(processed_b)

        normalized_a = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(flatten_a)
        normalized_b = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(flatten_b)

        L1_layer = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
        both = L1_layer([normalized_a, normalized_b])
        x = layers.Dense(self.dense_neurons, activation=self.activation_fct)(both)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.dense_neurons, activation=self.activation_fct)(x)
        x = layers.Dropout(self.dropout_rate)(x)
        prediction = layers.Dense(1, activation='sigmoid')(x)
        return prediction

