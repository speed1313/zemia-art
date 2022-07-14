"""
The script demonstrates a simple example of using ART with TensorFlow v1.x. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_mnist
import matplotlib.pyplot as plt
# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()



# Step 2: Create the model

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D


class TensorFlowModel(Model):
    """
    Standard TensorFlow model for unit testing.
    """

    def __init__(self):
        super(TensorFlowModel, self).__init__()
        self.conv1 = Conv2D(filters=4, kernel_size=5, activation="relu")
        self.conv2 = Conv2D(filters=10, kernel_size=5, activation="relu")
        self.maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format=None)
        self.flatten = Flatten()
        self.dense1 = Dense(100, activation="relu")
        self.logits = Dense(10, activation="linear")

    def call(self, x):
        """
        Call function to evaluate the model.

        :param x: Input to the model
        :return: Prediction of the model
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.logits(x)
        return x


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


model = TensorFlowModel()
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Step 3: Create the ART classifier

classifier = TensorFlowV2Classifier(
    model=model,
    loss_object=loss_object,
    train_step=train_step,
    nb_classes=10,
    input_shape=(28, 28, 1),
    clip_values=(0, 1),
)

# Step 4: Train the ART classifier

classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)
for i in range(5):
    noise_list = []
    epsilon_list = [j * 0.1 for j in range(15)]
    prediction_list = []
    for j in range(15):
        epsilon = j*0.1
        attack = FastGradientMethod(estimator=classifier, eps=epsilon)
        x_test_adv = attack.generate(x=x_test[i:i+1])
        noise_list.append(x_test[i] - x_test_adv[0])
        prediction = classifier.predict(x_test_adv[0:1])
        prediction_list.append(prediction)
        plt.subplot(3, 5, j+1)
        plt.imshow(x_test_adv[0])
        plt.viridis()
        plt.title(f"{np.argmax(prediction)}")
        plt.savefig(f"{i}_{np.argmax(y_test[i])}.png")
    plt.clf()
    for j in range(15):
        plt.subplot(3,5, j+1)
        plt.imshow(noise_list[j])
        plt.viridis()
        plt.title(f"{j}*0.1")
        plt.savefig(f"{i}_{np.argmax(y_test[i])}_noise.png")
    plt.clf()
    for j in range(10):
        plt.plot(epsilon_list, [np.array(prediction_list)[k][0][j] for k in range(len(epsilon_list))], '-o')
        plt.xlabel("epsilon")
        plt.ylabel("prediction")
        plt.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], loc='upper right', framealpha = 0.3)
    plt.savefig(f"{i}_{np.argmax(y_test[i])}_plot.png")

for i in range(5):
    noise_list = []
    epsilon_list = [j * 0.1 for j in range(15)]
    prediction_list = []
    random = np.random.randn(28,28,1)
    for j in range(15):
        epsilon = j*0.1
        noise = epsilon * random
        x_test_adv = x_test[i] + noise
        noise_list.append(noise)
        prediction = classifier.predict([x_test_adv])

        prediction_list.append(prediction)
        plt.subplot(3, 5, j+1)
        plt.imshow(x_test_adv)
        plt.viridis()
        plt.title(f"{np.argmax(prediction)}")
        plt.savefig(f"{i}_{np.argmax(y_test[i])}_random_ver.png")
    plt.clf()
    for j in range(15):
        plt.subplot(3,5, j+1)
        plt.imshow(noise_list[j])
        plt.viridis()
        plt.title(f"{j}*0.1")
        plt.savefig(f"{i}_{np.argmax(y_test[i])}_noise_random_ver.png")
    plt.clf()
    for j in range(10):
        plt.plot(epsilon_list, [np.array(prediction_list)[k][0][j] for k in range(len(epsilon_list))], '-o')
        plt.xlabel("epsilon")
        plt.ylabel("prediction")
        plt.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], loc='upper right', framealpha = 0.3)
    plt.savefig(f"{i}_{np.argmax(y_test[i])}_plot_random_ver.png")
    plt.clf()
