import tensorflow as tf

IMG_SIZE = (32, 32)
BATCH_SIZE = 32


def load_datasets():
    """
    Load and preprocess the CIFAR-10 datasets.

    Returns:
        train_dataset
        validation_dataset
        test_dataset
        class_names
    """

    ### Data augmentation (training only)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])

    ### Training dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        "data/cifar10/train",
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_dataset.class_names

    ### Validation dataset
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        "data/cifar10/train",
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    ### Test dataset
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        "data/cifar10/test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    ### Normalization
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x), y)
    )

    train_dataset = train_dataset.map(
        lambda x, y: (normalization_layer(x), y)
    )

    validation_dataset = validation_dataset.map(
        lambda x, y: (normalization_layer(x), y)
    )

    test_dataset = test_dataset.map(
        lambda x, y: (normalization_layer(x), y)
    )

    print("\nDataset Shapes:")

    for images, labels in train_dataset.take(1):
        print("Training batch:", images.shape)
        print("Training labels:", labels.shape)

    for images, labels in validation_dataset.take(1):
        print("Validation batch:", images.shape)
        print("Validation labels:", labels.shape)

    for images, labels in test_dataset.take(1):
        print("Test batch:", images.shape)
        print("Test labels:", labels.shape)

    # Performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset, class_names


if __name__ == "__main__":

    train_dataset, validation_dataset, test_dataset, class_names = load_datasets()

    print("\nDataset loaded successfully!")
    print("Classes:", class_names)
    print("Number of classes:", len(class_names))
    print("\nCIFAR-10 preprocessing completed successfully.")
