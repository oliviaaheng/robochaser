# Put new model here
import tensorflow as tf

class Finder(tf.keras.Model):
    
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        # Feature extraction
        self.resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(416, 416, 3))

        self.resize = tf.keras.layers.UpSampling2D(size=(32, 32))

        self.smallDetection = tf.keras.Sequential([
            tf.keras.layers.Conv2D(2048, (3, 3), padding='same', strides=(32, 32)),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.BatchNormalization(),
        ])

        self.upToMedium = tf.keras.layers.UpSampling2D(size=(32, 32))

        self.mediumDetection = tf.keras.Sequential([
            tf.keras.layers.Conv2D(2048, (3, 3), padding='same', strides=(16, 16)),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.BatchNormalization(),
        ])

        self.upToLarge = tf.keras.layers.UpSampling2D(size=(16, 16))

        self.largeDetection = tf.keras.Sequential([
            tf.keras.layers.Conv2D(2048, (3, 3), padding='same', strides=(8, 8)),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.BatchNormalization(),
        ])

        self.smallPred = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(64),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(5, activation='sigmoid')
        ])

        self.mediumPred = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(64),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(5, activation='sigmoid')
        ])

        self.largePred = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(64),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(5, activation='sigmoid')
        ])

        self.adder = tf.keras.layers.Add()

    def call(self, x):
        # Feature extraction
        x = self.resnet(x)
        x = self.resize(x)

        # Save large residual
        large_residual = x

        # Save medium residual
        medium_residual = x

        # Scale down to small detection
        small = self.smallDetection(x)

        # Upsample to medium size
        x = self.upToMedium(small)

        # Add back medium residual
        x = self.adder([x, medium_residual])

        # Medium detection
        medium = self.mediumDetection(x)

        # Upscale to large
        x = self.upToLarge(medium)

        # Add back large residual
        x = self.adder([x, large_residual])

        # Large detection
        large = self.largeDetection(x)

        large = self.largePred(large)
        medium = self.mediumPred(medium)
        small = self.smallPred(small)

        pred = self.adder([large, small, medium])

        return pred
    
