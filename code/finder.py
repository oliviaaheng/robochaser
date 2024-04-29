# Put new model here

import tensorflow as tf

class Finder(tf.keras.Model):
    
    def __init__(self, **kwargs):

        super().__init__(**kwargs)


        self.c1 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', strides=(16, 16))
        self.r1 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.n1 = tf.keras.layers.BatchNormalization()
        self.u1 = tf.keras.layers.UpSampling2D(size=(8, 8))

        self.c2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', strides=(8,  8))
        self.r2 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.n2 = tf.keras.layers.BatchNormalization()
        self.p2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.u2 = tf.keras.layers.UpSampling2D(size=(4, 4))

        self.c3 = tf.keras.layers.Conv2D(64, (5, 5), padding='same', strides=(2, 2))
        self.r3 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.n3 = tf.keras.layers.BatchNormalization()
        self.p3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.f4 = tf.keras.layers.Flatten()
        self.d4 = tf.keras.layers.Dense(256)
        self.r4 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.d5 = tf.keras.layers.Dense(64)
        self.r5 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.d6 = tf.keras.layers.Dense(5, activation='sigmoid')

    def call(self, x):
        x = self.c1(x)
        x = self.r1(x)
        x = self.n1(x)
        x = self.u1(x)

        x = self.c2(x)
        x = self.r2(x)
        x = self.n2(x)
        x = self.p2(x)
        x = self.u2(x)

        x = self.c3(x)
        x = self.r3(x)
        x = self.n3(x)
        x = self.p3(x)

        x = self.f4(x)
        x = self.d4(x)
        x = self.r4(x)

        x = self.d5(x)
        x = self.r5(x)

        x = self.d6(x)

        return x