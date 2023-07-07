import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import Model, layers


# Configure parameters
# TODO
# learning_rate = 0.001
# weight_decay = 0.0001
# batch_size = 256
# num_epochs = 800
# image_size = 72  # We'll resize input images to this size
# patch_size = 6  # Size of the patches to be extract from the input images
# num_patches = (image_size // patch_size) ** 2
# projection_dim = 64
# num_heads = 4
# transformer_units = [
#     projection_dim * 2,
#     projection_dim,
# ]  # Size of the transformer layers
# transformer_layers = 10
# mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier


# This function takes in an input tensor, number of hidden units, and dropout rate as parameters
def mlp(x, hidden_units, dropout_rate):
    # Iterate through the list of hidden units
    for units in hidden_units:
        # Use the tensorflow.keras.layers.Dense layer to create a fully connected neural network layer
        # with the given number of units and gelu activation function
        x = layers.Dense(units, activation=tf.nn.gelu)(x)

        # Apply the tensorflow.keras.layers.Dropout layer with the given dropout rate to help reduce overfitting
        x = layers.Dropout(dropout_rate)(x)

    # Return the final output tensor after passing through all hidden layers
    return x


# Defining a class named Patches that inherits from the tensorflow.keras.layers.Layer class
class Patches(layers.Layer):
    # Constructor to define the patch size for extracting patches from input image tensor
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    # The call method that extracts patches from the given input image tensor
    def call(self, images):
        # Get the batch size of the input images tensor
        batch_size = tf.shape(images)[0]

        # Extract patches from the input images tensor using the tensorflow function extract_patches
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        # Get the shape of the extracted patches tensor along the last dimension
        patch_dims = patches.shape[-1]

        # Reshape the extracted patches tensor to match the expected output shape and return it
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


# Define a function called plot_patches_raw that takes an integer sample index as input
def plot_patches_raw(x_train: any, sample_index: int, image_size: int, patch_size: int):
    # Set the size of the plot figure and choose a random training image from x_train tensor with given index
    plt.figure(figsize=(4, 4))
    image = x_train[np.random.choice(range(x_train.shape[sample_index]))]

    # Display the image on the plot
    plt.imshow(image.astype("uint8"))
    plt.axis("off")

    # Resize the image tensor, extract non-overlapping patches of specified size from it using the Patches layer,
    # and print out some details of the extracted patches tensor
    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]), size=(image_size, image_size)
    )
    patches = Patches(patch_size)(resized_image)
    print(f"Image size: {image_size} X {image_size}")
    print(f"Patch size: {patch_size} X {patch_size}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")

    # Determine number of patches to display in each row/column of subplot grid
    n = int(np.sqrt(patches.shape[1]))

    # Create a new plot figure and display each extracted patch as a separate sub-plot
    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches[sample_index]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")


# Define a new class named 'PatchEncoder' that inherits from the 'Layer' class
class PatchEncoder(layers.Layer):
    # Define constructor method that sets the number of patches and projection dimension size
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches

        # Define the dense (fully-connected) layer used for feature projection from a patch tensor
        self.projection = layers.Dense(units=projection_dim)

        # Define embedding layer which maps the position of the input patch in the input image to a high-dimensional space
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    # Define call method that performs forward pass through the layers of the 'PatchEncoder'
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)

        # Project the feature vectors with dense layer and add position embeddings via element-wise addition
        encoded = self.projection(patch) + self.position_embedding(positions)

        # Return the encoded feature vectors representing the locations of the input patches
        return encoded


# Function to create a Vision Transformer based classification model
def create_vit_classifier(
    x_train: any,
    input_shape: tuple,
    image_size: int,
    patch_size: int,
    num_patches: int,
    projection_dim: int,
    transformer_layers: int,
    num_heads: int,
    transformer_units: list,
    mlp_head_units: list,
    num_classes: int,
):
    """
    The code defines a function `create_vit_classifier` which takes several input arguments:

    - `x_train`: This variable represents the training dataset. It can be of any data type.
    - `input_shape`: This is the shape of the input tensor (excluding batch_size) that will be fed into the network. It is specified as a tuple.
    - `image_size`: This is the size of the image height/width in pixels.
    - `patch_size`: This is the size of the square patches that will be extracted from the images for processing.
    - `num_patches`: This is the number of patches that will be extracted from each input image.
    - `projection_dim`: This is the dimension of the projection space for the patch embeddings.
    - `transformer_layers`: This is the number of transformer layers to be applied in the model.
    - `num_heads`: This is the number of heads in the multi-head attention mechanism.
    - `transformer_units`: This is a list of integers representing the hidden layer sizes in the transformer blocks.
    - `mlp_head_units`: This is a list of integers representing the hidden layer sizes in the multi-layer perceptron (MLP) head.
    - `num_classes`: This is the number of output classes for classification.

    The purpose of this function is to create an image classification model based on the Vision Transformer architecture, also known as ViT. The function initializes and configures the various layers of the ViT model such as input layer, augmented layer, patch extraction layer, multiple transformer layers, MLP head, etc to build the complete ViT model.

    Once all the layers have been configured, the function returns the final Keras model that can be trained using the provided `x_train` dataset. By adjusting the hyperparameters for these arguments, you can customize the ViT model to suit the specific needs of your problem domain.
    """

    # Data Augmentation
    data_augmentation = keras.Sequential(
        [
            layers.Normalization(),
            layers.Resizing(image_size, image_size),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="data_augmentation",
    )

    # Compute the mean and the variance of the training data for normalization.
    data_augmentation.layers[0].adapt(x_train)

    # Define input shape of the image tensor
    inputs = layers.Input(shape=input_shape)

    # Augment the input data with random transformations to improve performance and dignal diversity
    augmented = data_augmentation(inputs)

    # Extract patches from the augmented images
    patches = Patches(patch_size)(augmented)

    # Encode the patches using multiple transformer blocks
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Normalize the output representation
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    # Flatten the feature map
    representation = layers.Flatten()(representation)

    # Apply dropout to reduce overfitting
    representation = layers.Dropout(0.5)(representation)

    # Classify outputs using a multi-layer perceptron head
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    # Generate the final output logits
    logits = layers.Dense(num_classes)(features)

    # Create and return a new Keras model with the provided inputs and the computed logits
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


def run_experiment(
    x_train: any,
    y_train: any,
    x_test: any,
    y_test: any,
    model: any,
    do_ckeckpoint: bool,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int,
):
    # Define an AdamW optimizer with learning rate and weight decay hyperparameters.
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    # Combine the optimizer with a SparseCategoricalCrossentropy loss metric
    # and add two evaluation metrics - SparseCategoricalAccuracy and SparseTopKCategoricalAccuracy.
    # The `SparseCategoricalAccuracy` metric calculates how often predictions match integer labels
    # and reports results as a percentage. The `SparseTopKCategoricalAccuracy` metric is similar to
    # `SparseCategoricalAccuracy`, but it only considers the top K predictions (in this case 5) instead
    # of just the most likely prediction.
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    # Create a callback that saves the model's weights whenever validation accuracy improves during training.
    # Specify the path where the checkpoint files will be saved.
    if do_ckeckpoint:
        checkpoint_filepath = "/tmp/checkpoint"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )

    # Train the model on the provided dataset `x_train` and `y_train`.
    # Use `batch_size` and `num_epochs` hyperparameters.
    # Split the data into 90% training and 10% validation sets.
    # Pass in the checkpoint callback to save best model weights during training.
    # Return the training history object.
    if do_ckeckpoint:
        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=0.1,
            callbacks=[checkpoint_callback],
        )
    else:
        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=0.1,
        )

    # Load the best model weights saved during training using the callback.
    # Evaluate the model's performance on the test dataset `x_test` and `y_test`.
    # Print the model's test accuracy and top 5 accuracy.
    # model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    # Return the training history object.
    return history, model


def extract_layers(model, num_layers):
    """
    Extract a specified number of layers from a trained TensorFlow model.

    Args:
        model (tf.keras.Model): The trained TensorFlow model.
        num_layers (int): The number of layers to extract from the model.

    Returns:
        new_model (tf.keras.Model): The new model with the specified number of layers.
    """
    # Ensure that the number of layers to extract is valid
    if num_layers <= 0 or num_layers > len(model.layers):
        raise ValueError(
            f"Invalid number of layers: {num_layers}. Must be between 1 and {len(model.layers)}"
        )

    # Get the input layer from the trained model
    input_layer = model.layers[0].input

    # Extract the specified number of layers
    output_layer = model.layers[num_layers - 1].output

    # Create a new model with the specified layers
    new_model = Model(inputs=input_layer, outputs=output_layer)

    return new_model


def plot_patches_with_model(
    x_train: any,
    model: any,
    sample_index: int,
    which_model: int,
    figsize: tuple,
    image_size: int,
    patch_size: int,
):
    plt.figure(figsize=figsize)
    image = x_train[np.random.choice(range(x_train.shape[sample_index]))] * 255
    plt.imshow(image.astype("uint8"))
    plt.axis("off")

    new_model = extract_layers(model, which_model)
    some_pred = new_model.predict(image.reshape((1, image.shape[0], image.shape[1], 3)))
    print(f"Prediction dim: {some_pred.shape}")
    some_pred = some_pred.reshape((144, some_pred.shape[2]))
    patch_importance_mat = np.argmax(some_pred, axis=1)
    patch_importance_mat = (
        np.exp(patch_importance_mat) / np.exp(patch_importance_mat).sum()
    )

    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]), size=(image_size, image_size)
    )
    patches = Patches(patch_size)(resized_image)
    print(f"Image size: {image_size} X {image_size}")
    print(f"Patch size: {patch_size} X {patch_size}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")

    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=figsize)
    for i, patch in enumerate(patches[sample_index]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        if patch_importance_mat[i] >= patch_importance_mat.mean():
            plt.title(f"W: {np.round(patch_importance_mat[i], 3)}", color="red")
        else:
            plt.title(f"W: {np.round(patch_importance_mat[i], 3)}")
        plt.axis("off")


def find_outlier(arr: np.ndarray, threshold: float = 3.0):
    assert arr.shape == (12, 12), "Input array must have shape (12, 12)"

    # Extract the middle 10x10 area
    middle_area = arr[1:11, 1:11]

    # Compute the mean and standard deviation of the middle area
    mean = middle_area.mean()
    std_dev = middle_area.std()

    # Define the outlier range
    lower_bound = mean - threshold * std_dev
    upper_bound = mean + threshold * std_dev

    # Iterate through the middle area and check for outliers
    for i in range(1, 11):
        for j in range(1, 11):
            if arr[i, j] < lower_bound or arr[i, j] > upper_bound:
                return (i, j)

    # If no outlier is found, return None
    return None


def display_image_with_label_and_rectangle(
    img: np.ndarray, model: tf.keras.Model, coords: tuple, w: int, h: int
):
    assert img.shape == (224, 224, 3), "Input image must have shape (224, 224, 3)"

    # Prepare the input data for the model
    input_data = np.expand_dims(img, axis=0)

    # Predict the label for the input image
    predictions = model.predict(input_data)
    predicted_label = np.argmax(predictions, axis=1)

    # Convert the label to string
    label_str = str(predicted_label[0])

    # Set the font, color, and size for the label text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 0, 0)  # Blue
    font_thickness = 2

    # Put the label text on the top left corner of the image
    cv2.putText(img, label_str, (5, 30), font, font_scale, font_color, font_thickness)

    # Draw the rectangle using the provided coordinates, width, and height
    x, y = coords
    cv2.rectangle(img, (x, y), (x + w, y + h), font_color, font_thickness)

    # Display the image using OpenCV
    cv2.imshow(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()