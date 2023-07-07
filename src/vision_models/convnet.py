import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical

class ConvNet:
    def __init__(self, input_dim, output_dim, base_model_name, model_name):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_model_name = base_model_name
        self.model_name = model_name
        self.base_model = self._get_base_model()

    def _get_base_model(self):
        if self.base_model_name == "VGG16":
            base_model = tf.keras.applications.VGG16(include_top=False, input_shape=self.input_dim, weights='imagenet')
        elif self.base_model_name == "VGG19":
            base_model = tf.keras.applications.VGG19(include_top=False, input_shape=self.input_dim, weights='imagenet')
        elif self.base_model_name == "ResNet50":
            base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=self.input_dim, weights='imagenet')
        elif self.base_model_name == "ResNet101":
            base_model = tf.keras.applications.ResNet101(include_top=False, input_shape=self.input_dim, weights='imagenet')
        else:
            raise ValueError(f"Invalid base model name: {self.base_model_name}")

        return base_model

    def train(self, X, Y, validation_split=0.2, learning_rate=0.0001, batch_size=32, epochs=10):
        for layer in self.base_model.layers:
            layer.trainable = False

        x = GlobalAveragePooling2D()(self.base_model.output)
        x = Dense(512, activation='relu')(x)
        output = Dense(self.output_dim, activation='softmax')(x)
        model = Model(inputs=self.base_model.input, outputs=output)

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=CategoricalCrossentropy(), metrics=['accuracy'])
        history = model.fit(X, Y, validation_split=validation_split, batch_size=batch_size, epochs=epochs)

        return model, history