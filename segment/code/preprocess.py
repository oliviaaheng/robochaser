import tensorflow as tf

def getData():
    # Set the path to your directories
    train_dir = '/Users/juliandhanda/Documents/Comp Sci/Car Chaser/robochaser/segment/toycar-detection/train'
    val_dir = '/Users/juliandhanda/Documents/Comp Sci/Car Chaser/robochaser/segment/toycar-detection/valid'
    test_dir = '/Users/juliandhanda/Documents/Comp Sci/Car Chaser/robochaser/segment/toycar-detection/test'

    # Parameters
    batch_size = 8
    img_height = 640
    img_width = 640

    # Load the training data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Load the validation data
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Load the test data
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

def main():
    train, val, test = getData()

if __name__ == '__main__':
    main()