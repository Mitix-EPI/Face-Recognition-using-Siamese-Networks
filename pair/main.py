from sklearn.datasets import fetch_lfw_pairs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import csv
from siamese_network import SiameseNetwork
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def visualize(image0, image1, label, proba=None):
    # Plot the images
    plt.imshow(np.concatenate([image0, image1], axis=1), cmap='gray')
    plt.title("Label: {}".format(label))
    if proba is not None:
        plt.suptitle("Proba: {:.2f}".format(proba))
    plt.show()

lfw_pairs_train = fetch_lfw_pairs(subset="train", resize=0.4)
lfw_pairs_test = fetch_lfw_pairs(subset="test", resize=0.4)
classes = list(lfw_pairs_train.target_names)


X = lfw_pairs_train.pairs
X = X.astype("float32")
y = lfw_pairs_train.target.astype("float32")

X_test = lfw_pairs_test.pairs
X_test = X_test.astype("float32")
y_test = lfw_pairs_test.target.astype("float32")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

IMG_SHAPE = (X_train.shape[2], X_train.shape[3], 1) # channel=1 because grayscale image
# IMG_SHAPE = (X_train.shape[2], X_train.shape[3], 3) # ! Only if you want to run fine-tunning

##### Data Augmentation

nb_new_data = 500

# Flip horizontally images
augmented_X_train_1 = X_train[:nb_new_data, 0]
augmented_X_train_2 = X_train[:nb_new_data, 1]
augmented_y_train = y_train[:nb_new_data]

datagen = ImageDataGenerator(
    zoom_range=[0.9,0.9],
    horizontal_flip=True,
)
datagen.fit(augmented_X_train_1.reshape(augmented_X_train_1.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], 1))
datagen.fit(augmented_X_train_2.reshape(augmented_X_train_2.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], 1))

data_generator_1 = datagen.flow(augmented_X_train_1.reshape(augmented_X_train_1.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], 1), shuffle=False, batch_size=1)
data_generator_2 = datagen.flow(augmented_X_train_2.reshape(augmented_X_train_2.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], 1), shuffle=False, batch_size=1)

X_train_aug_1 = [data_generator_1.next() for i in range(0, nb_new_data)]
X_train_aug_2 = [data_generator_2.next() for i in range(0, nb_new_data)]

augmented_X_train_1 = np.asarray(X_train_aug_1).reshape(nb_new_data, IMG_SHAPE[0], IMG_SHAPE[1])
augmented_X_train_2 = np.asarray(X_train_aug_2).reshape(nb_new_data, IMG_SHAPE[0], IMG_SHAPE[1])

augmented_X_train = np.array([augmented_X_train_1, augmented_X_train_2])
arr = np.transpose(augmented_X_train, (1, 0, 2, 3))

# reshape to the desired shape
augmented_X_train = arr.reshape(nb_new_data, 2, IMG_SHAPE[0], IMG_SHAPE[1])

# Concatenate the augmented data with the original training data
X_train = np.concatenate([X_train, augmented_X_train])
y_train = np.concatenate([y_train, augmented_y_train])

######

X_train /= 255
X_val /= 255
X_test /= 255

img1_train = X_train[:, 0]
img2_train = X_train[:, 1]

print(img1_train.shape)

# ! Only if you want to run fine-tunning
# img1_train = np.stack([img1_train], axis=-1)
# img1_train = np.repeat(img1_train, 3, axis=-1)
# img2_train = np.stack([img2_train], axis=-1)
# img2_train = np.repeat(img2_train, 3, axis=-1)

print(img1_train.shape)

img1_val = X_val[:, 0]
img2_val = X_val[:, 1]

# ! Only if you want to run fine-tunning
# img1_val = np.stack([img1_val], axis=-1)
# img1_val = np.repeat(img1_val, 3, axis=-1)
# img2_val = np.stack([img2_val], axis=-1)
# img2_val = np.repeat(img2_val, 3, axis=-1)

img1_test = X_test[:, 0]
img2_test = X_test[:, 1]

# ! Only if you want to run fine-tunning
# img1_test = np.stack([img1_test], axis=-1)
# img1_test = np.repeat(img1_test, 3, axis=-1)
# img2_test = np.stack([img2_test], axis=-1)
# img2_test = np.repeat(img2_test, 3, axis=-1)

def plot_training(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

def plot_training_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.show()

test_batch_size = [32]
test_epochs = [30]
test_lr = [1e-3, 1e-4]
test_activation_fct = ["relu"]
test_cnn_filters = [ [64, 64, 128], [32, 64, 64, 128], [16, 32, 64] ]
test_fine_tunning = [False]
test_dense_neurons = [500, 1024, 2048]
test_dropout_rate = [0.25]

results = []

def write_to_csv(results, filename):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(results)

for batch_size in test_batch_size:
    for epochs in test_epochs:
        for lr in test_lr:
            for activation_fct in test_activation_fct:
                for cnn_filters in test_cnn_filters:
                    for fine_tunning in test_fine_tunning:
                        for dense_neurons in test_dense_neurons:
                            for dropout_rate in test_dropout_rate:

                                sn = SiameseNetwork(IMG_SHAPE, cnn_filters, epochs, lr, batch_size, dense_neurons, activation_fct, fine_tunning, dropout_rate)

                                sn.get_siamese_network()
                                history = sn.train([img1_train, img2_train], y_train, [img1_val, img2_val], y_val)
                                
                                train_accuracy = history.history['accuracy'][-1]
                                val_acc = history.history['val_accuracy']
                                best_epoch_index = val_acc.index(max(val_acc))
                                best_val_acc = val_acc[best_epoch_index]
                                
                                write_to_csv([train_accuracy, best_val_acc, best_epoch_index + 1, IMG_SHAPE, batch_size, epochs, lr, activation_fct, cnn_filters, fine_tunning, dense_neurons, dropout_rate], "./parameter_test.csv")

# preds = sn.predict([img1_test, img2_test])

# pred0 = preds[0][0]
# visualize(img1_test[0], img2_test[0], classes[int(y_test[0])], pred0)
# pred1 = preds[-1][0]
# visualize(img1_test[-1], img2_test[-1], classes[int(y_test[-1])], pred1)

