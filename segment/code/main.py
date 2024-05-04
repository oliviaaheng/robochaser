import tensorflow as tf
from preprocess import getData

def getSimpleModel():
    return tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Using sigmoid since it's a binary classification
])

def getComplexModel():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input((640, 640, 3)),
        tf.keras.layers.Rescaling(1./255),

        tf.keras.layers.Conv2D(32, (5, 5), padding='same', strides=(2, 2)),
        tf.keras.layers.LeakyReLU(negative_slope=0.02),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),

        tf.keras.layers.Conv2D(64, (5, 5), padding='same', strides=(2, 2)),
        tf.keras.layers.LeakyReLU(negative_slope=0.02),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),

        tf.keras.layers.Conv2D(128, (3, 3), padding='same', strides=(2, 2)),
        tf.keras.layers.LeakyReLU(negative_slope=0.02),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64),
        tf.keras.layers.LeakyReLU(negative_slope=0.02),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(64),
        tf.keras.layers.LeakyReLU(negative_slope=0.02),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(1, activation='sigmoid')  # Using sigmoid since it's a binary classification
    ])

def main():
    train_ds, val_ds, test_ds = getData()

    model = getSimpleModel()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',  # Suitable for binary classification
        metrics=['accuracy']
    )

    x = tf.ones((1, 640, 640, 3))
    model.call(x)
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10  # You can adjust the number of epochs based on your specific needs
    )

    test_loss, test_acc = model.evaluate(test_ds)
    print("Test accuracy:", test_acc)

if __name__ == '__main__':
    main()