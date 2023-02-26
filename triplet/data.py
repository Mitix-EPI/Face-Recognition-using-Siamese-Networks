import matplotlib.pyplot as plt
import numpy as np
import random

def visualize(image0, image1, proba=None):
    # Plot the images
    plt.imshow(np.concatenate([image0, image1], axis=1), cmap='gray')
    if proba is not None:
        plt.suptitle("Distance: {:.2f}".format(proba))
    plt.show()


def triplet_data_generator(X, y, image_shape, max_size=None):
    total_size = max_size if max_size is not None else max(y)
    triplets = np.zeros((3, total_size, image_shape[0], image_shape[1], 3))
    idx_to_delete = []

    for i in range(total_size):
        idx = i
        anchor_image = X[idx]
        anchor_label = y[idx]

        same_person = np.where(y == anchor_label)[0]
        if len(same_person) > 1:
            idx_i = np.isin(same_person, idx)
            same_person = np.delete(same_person, idx_i)
            positive_index = np.random.choice(same_person)
            negative_index = np.random.choice(np.where(y != anchor_label)[0])
            positive_image = X[positive_index]
            negative_image = X[negative_index]
            # visualize(anchor_image, positive_image)
            # visualize(anchor_image, negative_image)
            triplets[0][i] = anchor_image
            triplets[1][i] = positive_image
            triplets[2][i] = negative_image
        else:
            idx_to_delete.append(i)
        # print("{percentage:.2f}%".format(percentage = (i * 100) / max(y)))

    triplets = np.delete(triplets, idx_to_delete, axis=1) # Remove people that doesn't have other photos of themself
    
    tmp = np.array([triplets[0,:], triplets[1,:], triplets[2,:]])
    tmp = np.transpose(tmp, (1, 0, 2, 3, 4))
    return tmp

def get_batch(train_triplet, batch_size):
    num_batches = len(train_triplet) // batch_size

    for i in range(num_batches):
        batch = train_triplet[i * batch_size : (i + 1) * batch_size]
        anchor_imgs = batch[:, 0]
        positive_imgs = batch[:, 1]
        negative_imgs = batch[:, 2]
        targets = np.zeros(batch_size) # or generate the appropriate targets
        yield [anchor_imgs, positive_imgs, negative_imgs], targets