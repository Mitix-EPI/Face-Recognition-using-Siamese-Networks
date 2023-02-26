import tensorflow as tf
from tensorflow.keras import layers, metrics
from tensorflow.keras.applications import ResNet152, Xception
import numpy as np
from tensorflow.keras import layers, models
from keras.regularizers import l2
from scipy import spatial

tf.random.set_seed(42)

# From Kaggle Class
class DistanceLayer(layers.Layer):
    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(tf.abs(anchor - positive)), -1)
        an_distance = tf.reduce_sum(tf.square(tf.abs(anchor - negative)), -1)
        return (ap_distance, an_distance)

class SiameseNetwork():

    def __init__(self,
                 IMG_SHAPE,
                 cnn_filters=[64,64,128],
                 epochs=15,
                 lr=1e-4,
                 batch_size=32,
                 fine_tunning=False,
                 dropout_rate=0.3,
                 embedding_dim=64) -> None:
        self.IMG_SHAPE = IMG_SHAPE
        self.cnn_filters = cnn_filters
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.fine_tunning = fine_tunning
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.model = None

    def get_embeddings(self):
        if self.fine_tunning is False:
            embedding_model = self._get_embeddings_from_scratch()
        else:
            embedding_model = self._get_fine_tunning_embeddings()
        return embedding_model

    def get_siamese_distance_siamese_network(self):
        anchor_input = layers.Input(shape=self.IMG_SHAPE, name='anchor_input')
        positive_input = layers.Input(shape=self.IMG_SHAPE, name='positive_input')
        negative_input = layers.Input(shape=self.IMG_SHAPE, name='negative_input')

        embedding_model = self.get_embeddings()

        # Generate the embeddings for the anchor, positive, and negative inputs
        anchor_embeddings = embedding_model(anchor_input)
        positive_embeddings = embedding_model(positive_input)
        negative_embeddings = embedding_model(negative_input)

        distances = DistanceLayer()(anchor_embeddings, positive_embeddings, negative_embeddings)

        self.model = models.Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances, name="Embedding")
        return self.model

    def train(self, generator):
        steps_per_epoch = self.batch_size
        history = self.model.fit(generator, steps_per_epoch=steps_per_epoch, batch_size=self.batch_size, epochs=self.epochs)
        return history
    
    def get_test_model(self, embedding_layer=None):
        if embedding_layer is None:
            embedding_layer = self.model.layers[3]

        anchor_input = tf.keras.Input(shape=self.IMG_SHAPE)
        anchor_embeddings = embedding_layer(anchor_input)
        embedding_model = models.Model(inputs=anchor_input, outputs=anchor_embeddings)
        return embedding_model

    def test(self, X_test, y_test, embedding_layer=None):
        embedding_model = self.get_test_model(embedding_layer)

        embedding1 = embedding_model.predict(np.array(X_test[:, 0]))
        embedding2 = embedding_model.predict(np.array(X_test[:, 1]))

        similarities = []
        for i in range(len(embedding1)):
            similarity = 1 - spatial.distance.cosine(embedding1[i], embedding2[i])
            similarities.append(similarity)

        similarities = np.array(similarities)
        return similarities

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
        model.add(layers.Dense(self.embedding_dim, activation=None))

        return model

    def _get_embeddings_from_scratch(self):
        return self._get_CNN()

    def _get_fine_tunning_embeddings(self):
        input = layers.Input(shape=self.IMG_SHAPE)
        # Xception
        pretrained_model = Xception(
            input_shape=self.IMG_SHAPE,
            weights='imagenet',
            include_top=False,
            pooling='avg',
        )

        for i in range(len(pretrained_model.layers)-27):
            pretrained_model.layers[i].trainable = False

        processed = pretrained_model(input)

        flatten = layers.Flatten()(processed)
        x = layers.Dense(512, activation='relu')(flatten)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(self.embedding_dim, activation="relu")(x)
        embeddings = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
        siamese_network = models.Model(inputs=input, outputs=embeddings, name="Embedding")
        return siamese_network


# From Kaggle class
class SiameseModel(models.Model):
    # Builds a Siamese model based on a base-model
    def __init__(self, siamese_network, margin=1.0):
        super(SiameseModel, self).__init__()
        
        self.margin = margin
        self.siamese_network = siamese_network
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)
            
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]
