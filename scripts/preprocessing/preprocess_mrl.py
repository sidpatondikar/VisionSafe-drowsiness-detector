import tensorflow as tf
import os

IMG_SIZE = 64
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

DATA_DIR = "data"
SPLITS = ['train', 'val', 'test']
LABELS = {'awake': 1, 'sleepy': 0}

def get_image_paths_and_labels(split):
    image_paths, labels = [], []
    for category, label in LABELS.items():
        folder = os.path.join(DATA_DIR, split, category)
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath):
                image_paths.append(fpath)
                labels.append(label)
    return image_paths, labels

def preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def create_dataset(split):
    paths, labels = get_image_paths_and_labels(split)
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

if __name__ == "__main__":
    for split in SPLITS:
        ds = create_dataset(split)
        print(f"{split} dataset created.")
        for batch_images, batch_labels in ds.take(1):
            print(f"{split} batch shape: {batch_images.shape}, labels: {batch_labels[:5].numpy()}")
