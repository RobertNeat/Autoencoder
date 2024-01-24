# Autoencoders

This Python notebook demonstrates the implementation and visualization of autoencoders to compress and reconstruct images. Autoencoders consist of an encoder, which compresses information into a set of hyperparameters, and a decoder, which reconstructs the information from the encoded form.

## Dataset

The code uses the MNIST dataset, consisting of hand-written digit images. The dataset is loaded, preprocessed, and scaled for further analysis.

```python
# loading the dataset
import numpy as np
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.expand_dims(X_train, axis=-1)
X_train_scaled = (X_train/255).copy()

```

# Autoencoder Structure

The structure of the autoencoder is defined with specific layers for both the encoder and decoder. Various hyperparameters and activation functions are utilized.

````python
# defining the autoencoder structure
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Input, Reshape, UpSampling2D, BatchNormalization, GaussianNoise
from keras.models import Model
from keras.optimizers import Adam

act_func = 'selu'
aec_dim_num = 2
encoder_layers = [GaussianNoise(1),
                  BatchNormalization(),
                  Conv2D(32, (7,7),padding='same', activation=act_func),
                  MaxPool2D(2,2),
                  BatchNormalization(),
                  Conv2D(64, (5,5),padding='same', activation=act_func),
                  MaxPool2D(2,2),
                  BatchNormalization(),
                  Conv2D(128, (3,3),padding='same', activation=act_func),
                  GlobalAveragePooling2D(),
                  Dense(aec_dim_num, activation='tanh')]

decoder_layers = [Dense(128, activation=act_func),
                  BatchNormalization(),
                  Reshape((1,1,128)),
                  UpSampling2D((7,7)),
                  Conv2D(32, (3,3), padding='same', activation=act_func),
                  BatchNormalization(),
                  UpSampling2D((2,2)),
                  Conv2D(32, (5,5),padding='same', activation=act_func),
                  BatchNormalization(),
                  UpSampling2D((2,2)),
                  Conv2D(32, (7,7),padding='same', activation=act_func),
                  BatchNormalization(),
                  Conv2D(1, (3,3),padding='same', activation='sigmoid')]

lrng_rate = 0.0002
tensor = input_aec = input_encoder = Input(X_train.shape[1:])

for layer in encoder_layers:
    tensor = layer(tensor)
output_encoder = tensor
dec_tensor = input_decoder = Input(output_encoder.shape[1:])
for layer in decoder_layers:
    tensor = layer(tensor)
    dec_tensor = layer(dec_tensor)
output_aec = tensor
output_decoder = dec_tensor
autoencoder = Model(inputs=input_aec, outputs=output_aec)
encoder = Model(inputs=input_encoder, outputs=output_encoder)
decoder = Model(inputs=input_decoder, outputs=dec_tensor)
autoencoder.compile(optimizer=Adam(lrng_rate), loss='binary_crossentropy')
autoencoder.fit(x=X_train, y=X_train, epochs=5, batch_size=256)


# Visualizing Hyperparameters

The notebook includes visualizations to represent learning data in the hyperparameters dimension.

```python
# representing learning data in the hyperparameters dimension
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(20, 16))
for i in range(10):
    digits = y_train == i
    needed_imgs = X_train[digits, ...]
    preds = encoder.predict(needed_imgs)
    ax.scatter(preds[:, 0], preds[:, 1])
ax.legend(list(range(10)))
````

# Generating Images Based on Hidden Hyperparameters

The code generates images based on different hidden hyperparameter values.

```python
# generating the images based on the different hidden hyperparameters value
num = 15
limit = 0.6
step = limit * 2 / num
fig, ax = plt.subplots(num, num, figsize=(20, 16))
X_vals = np.arange(-limit, limit, step)
Y_vals = np.arange(-limit, limit, step)
for i, x in enumerate(X_vals):
    for j, y in enumerate(Y_vals):
        test_in = np.array([[x, y]])
        output = decoder.predict(x=test_in)
        output = np.squeeze(output)
        ax[-j-1, i].imshow(output, cmap='jet')
        ax[-j-1, i].axis('off')

```

# Image Denoising

The notebook demonstrates image denoising using autoencoders. It adds noise to test photos, then uses the trained autoencoder to remove the noise.

```python
# Initialize noisy_test_photos
noisy_test_photos = X_test[10:20, ...].copy()

# generating noise images
test_photos = X_test[10:20, ...].copy()
mask = np.random.randn(*test_photos.shape)
white = mask > 1
black = mask < -1
noisy_test_photos[white] = 255
noisy_test_photos[black] = 0
noisy_test_photos = noisy_test_photos.astype('float32') / 255  # Convert to float

# removing noise from images using autoencoder
def show_pictures(arrs):
    arr_cnt = arrs.shape[0]
    fig, axes = plt.subplots(1, arr_cnt, figsize=(5 * arr_cnt, arr_cnt))
    for axis, pic in zip(axes, arrs):
        axis.imshow(pic.squeeze(), cmap='gray')
cleaned_images = autoencoder.predict(noisy_test_photos/255) * 255
show_pictures(test_photos)
show_pictures(noisy_test_photos)
show_pictures(cleaned_images)
```

Generated images are displayed to show the original, noisy, and cleaned versions.

# Important
The branch containing fashion is the same manipulations but for different image dataset.

## Launching Project

To run the project, you have a couple of options:

### Using Google Colab via Gist

Access the project through Google Colab using the Gist website. You can import the necessary data from the GitHub project resources. Use the following Gist link: [gist link here](https://gist.github.com/RobertNeat/91b9911ee45190680f4c815164cdebc9)

### Running Locally

If you prefer to run the project on your local machine, follow these steps:

1. **Clone the Repository**: Download the repository branch from GitHub.
2. **Local Environment**:
   - **DataSpell or PyCharm**: Open the project using DataSpell or PyCharm by JetBrains.
   - **Spyder IDE**: Alternatively, you can use Spyder IDE to work with the project.
3. **Dataset Requirements**:
   - Ensure that the dataset files are available and stored inside your project directory. This step is crucial to prevent any issues related to missing data.

Running the project locally allows you to explore the code and execute it in your preferred Python environment. If you encounter any problems, make sure to check the dataset's presence in your project directory.
