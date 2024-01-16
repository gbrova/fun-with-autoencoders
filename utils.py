import datasets
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm


import tensorflow as tf
from keras.datasets import mnist


from PIL import Image


def preprocess_image(img, resized_size):
    # convert PIL.Image to tensor
    img = img.resize((resized_size, resized_size))
    img = tf.keras.utils.img_to_array(img, data_format=None, dtype=None)
    img = img / 256
    return img
    

def get_preprocessed_emoji(resized_size):
    
    dataset = datasets.load_dataset("valhalla/emoji-dataset")
    
    x_train = np.array([preprocess_image(img, resized_size) for img in tqdm(dataset["train"][:2000]["image"])])
    x_test = np.array([preprocess_image(img, resized_size) for img in tqdm(dataset["train"][2000:]["image"])])

    return x_train, x_test


def get_preprocessed_mnist(resized_size):
    (mnist_train, _), (mnist_test, _) = mnist.load_data()
    
    
    # nb: .convert('RGB') to force 3 channels, just so the shapes are the same.
    x_train = np.array([preprocess_image(Image.fromarray(img).convert('RGB'), resized_size) for img in tqdm(mnist_train[:30_000])])
    x_test = np.array([preprocess_image(Image.fromarray(img).convert('RGB'), resized_size) for img in tqdm(mnist_test)])

    return x_train, x_test



def plot_loss(history):
    # Obtain the training and validation loss values
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Plot the loss curves
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


# Utility functions
def baseline_recover(item, equivalent_downsample):
    original_size = item.shape[0]
    img = tf.keras.utils.array_to_img(item, data_format=None, scale=True, dtype=None)
    img = img.resize((equivalent_downsample, equivalent_downsample))
    img = img.resize((original_size, original_size))
    img = tf.keras.utils.img_to_array(img, data_format=None, dtype=None)
    img /= 256
    return img
    

def view_images(images):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(12, 8))

    for i, img in enumerate(images):
        # img = Image.open(images[i])
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def evaluate_autoencoder(autoencoder, x_eval, idxs=None):
    mse = tf.keras.losses.MeanSquaredError()
    recovered = autoencoder.predict(x_eval)

    # all of our autoencoders will have precisely two Sequential models, the encoder and decoder.
    assert len(autoencoder.layers) == 2
    encoder = autoencoder.layers[0] 
    
    bottleneck_layer_size = np.prod(encoder.layers[-1].output_shape[1:])
    equivalent_downsample = int((bottleneck_layer_size / 3) ** 0.5)  # use as many "numbers" (forget precision for now) as the bottleneck layer.

    recovered_downsample = np.array([baseline_recover(img, equivalent_downsample) for img in x_eval])

    print(f"Bottleneck layer size: {bottleneck_layer_size}, equivalent_downsample={equivalent_downsample}")
    
    print(f"Recovered MSE: {mse(recovered, x_eval).numpy()}")
    print(f"Baseline MSE:  {mse(recovered_downsample, x_eval).numpy()}")

    # see some images
    if idxs is None:
        idxs = [1,4, 12, 222, 314, 612]  # [12, 143, 222, 314, 612]
    print("Originals:")
    view_images(x_eval[idxs])
    
    print("AE Recovered:")
    view_images(recovered[idxs])
    
    print("Downsample/Upsample baseline:")
    view_images(recovered_downsample[idxs])

