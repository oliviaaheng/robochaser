import os
import tensorflow as tf
import numpy as np
import random
import math

def getHyperparams():
    epochs = 10
    batchSize = 64

    trainingRate = 0.01

    loss = 'mse'

    return [epochs, batchSize, trainingRate, loss]

def getLocatorModel(rate, loss):
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
        loss=loss, 
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

    locator = getLocatorModel(params[2], params[3])

    trainStats = locator.fit(x_train, y_train,
                    batch_size=params[1],  # Specify your desired batch size
                    epochs=params[0],
                    validation_data=(x_val, y_val))
    
    locator.evaluate(x_test, y_test)

    locator.save('locator')

    return 0

if __name__ == '__main__':
    main()
        