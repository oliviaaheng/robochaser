import os
import tensorflow as tf
import numpy as np
import random
import math

def getHyperparams():
    epochs = 10
    batchSize = 64

    trainingRate = 0.01

    return [epochs, batchSize, trainingRate]

def customLoss(true, pred):
    # Isolate the first element and the remaining elements
    first_element_true = true[:, 0]
    first_element_pred = pred[:, 0]
    remaining_true = true[:, 1:]
    remaining_pred = pred[:, 1:]

    # Compute MSE for the first element
    mse_first = tf.reduce_mean(tf.square(first_element_true - first_element_pred))

    # Compute MSE for the remaining four elements
    mse_remaining = tf.reduce_mean(tf.square(remaining_true - remaining_pred), axis=1)

    # Multiply the MSE of the remaining elements by the first element of the label
    # If the label is all zeros (no image detected), the loss will only be dependent on the
    # confidence score, the first elements. Loss will be higher if a high confidence score 
    # is given to a zero nothing label.
    custom_loss = first_element_true * mse_remaining + mse_first

    # Average this custom loss over the batch
    return tf.reduce_mean(custom_loss)

def getLocatorModel(rate):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(5, (1, 1), activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=rate), 
        loss=customLoss, 
        metrics=['accuracy'],
    )

    return model

def main():
    # Get the data

    x_train = None
    y_train= None
    x_val = None
    y_val = None
    x_test = None
    y_test = None

    params = getHyperparams()

    locator = getLocatorModel(params[2])

    trainStats = locator.fit(x_train, y_train,
                    batch_size=params[1],  # Specify your desired batch size
                    epochs=params[0],
                    validation_data=(x_val, y_val))
    
    locator.evaluate(x_test, y_test)

    locator.save('locator')

    return 0

if __name__ == '__main__':
    main()
        