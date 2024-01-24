# Fashion MNIST Autoencoder

This Python notebook implements an autoencoder using only dense layers to differentiate the images in the Fashion MNIST dataset. The code includes the addition of Gaussian noise to the input data, securing against out-of-range values, defining the autoencoder structure, and visualizing the results.

## Getting Started

### Prerequisites

Make sure you have the necessary dependencies installed. You can install them using the following:

```python
pip install numpy keras matplotlib
```

### Dataset

The code uses the Fashion MNIST dataset. It is loaded into X_train, y_train, X_test, and y_test.

```python
import numpy as np
from keras.datasets import fashion_mnist

# Loading the dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

```

## Adding Gaussian Noise

Gaussian noise is added to the input data to create x_train_noisy and x_test_noisy.

```python
x_train_noisy = X_train + np.random.normal(loc=0.0, scale=0.5, size=X_train.shape)
x_test_noisy = X_test + np.random.normal(loc=0.0, scale=0.5, size=X_test.shape)

# Clipping values to the range [0, 1]
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

```

## Autoencoder Architecture

The autoencoder structure is defined using dense layers. The encoder and decoder layers are specified separately.

```python
from keras.layers import Flatten, Dense, Input, Reshape
from keras.models import Model
from keras.optimizers import Adam

act_func = 'selu'
aec_dim_num = 2

encoder_layers = [
    Flatten(input_shape=(28, 28, 1)),
    Dense(256, activation=act_func),
    Dense(128, activation=act_func),
    Dense(32, activation=act_func)
]

decoder_layers = [
    Dense(128, activation=act_func),
    Dense(256, activation=act_func),
    Dense(784, activation='sigmoid'),  # 28x28=784
    Reshape((28, 28, 1))
]

# ...

```

## Training the Autoencoder

The autoencoder is compiled and trained using the noisy input and original images.

```python
autoencoder.compile(optimizer=Adam(lr=lrng_rate), loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, X_train, epochs=30, batch_size=256, shuffle=True, validation_data=(x_test_noisy, X_test))
```

## Visualization

The results are visualized by generating denoised images from the noisy ones.

```python
import matplotlib.pyplot as plt

decoded_imgs = autoencoder.predict(x_test_noisy)

# Plotting original, noisy, and denoised images
# ...

```

## Hyperparameters Dimension Visualization

The hyperparameters dimension is visualized by scatter plotting the encoded representations of different digits.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(20, 16))

for i in range(10):
    digits = y_train == i
    needed_imgs = X_train[digits, ...]
    preds = encoder.predict(needed_imgs)
    ax.scatter(preds[:, 0], preds[:, 1])

ax.legend(list(range(10)))
plt.show()

```

## Launching Project

To run the project, you have a couple of options:

### Using Google Colab via Gist

Access the project through Google Colab using the Gist website. You can import the necessary data from the GitHub project resources. Use the following Gist link: [gist link here](https://gist.github.com/RobertNeat/2cf96267081f14bc321af78445a0a18e)

### Running Locally

If you prefer to run the project on your local machine, follow these steps:

1. **Clone the Repository**: Download the repository branch from GitHub.
2. **Local Environment**:
   - **DataSpell or PyCharm**: Open the project using DataSpell or PyCharm by JetBrains.
   - **Spyder IDE**: Alternatively, you can use Spyder IDE to work with the project.
3. **Dataset Requirements**:
   - Ensure that the dataset files are available and stored inside your project directory. This step is crucial to prevent any issues related to missing data.

Running the project locally allows you to explore the code and execute it in your preferred Python environment. If you encounter any problems, make sure to check the dataset's presence in your project directory.
