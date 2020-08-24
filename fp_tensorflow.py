import tensorflow as tf
import tensorflow.keras.backend as K


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Reshape,
    Flatten,
    Dropout,
    BatchNormalization,
    Activation,
    ZeroPadding2D,
    LeakyReLU,
    UpSampling2D,
    Conv2D,
    Convolution2D,
    MaxPooling2D,
    Concatenate,
    GaussianNoise,
    GaussianDropout,
    Lambda,
    GlobalAveragePooling2D,
)

### parse functions


def _parse(example_proto, fp_dim=(56, 56, 6)):
    # Create a description of the features.
    feature_description = {
        "floorplan": tf.io.FixedLenFeature(
            fp_dim, tf.float32, default_value=tf.zeros(fp_dim, tf.float32)
        ),
        "plan_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "year": tf.io.FixedLenFeature([], tf.int64, default_value=-1),  # 0~9
        "sido": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "norm_area": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "num_rooms": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "num_baths": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }

    # Parse the input tf.Example proto using the dictionary above.
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    return parsed_example


def _parse_pair(
    example_proto, feature1="floorplan", feature2="year", fp_dim=(56, 56, 6)
):
    parsed_example = _parse(example_proto, fp_dim)
    return parsed_example[feature1], parsed_example[feature2]


def _parse_single(example_proto, feature_id="floorplan", fp_dim=(56, 56, 6)):
    parsed_example = _parse(example_proto, fp_dim)
    return parsed_example[feature_id]


_parse_pair_56 = lambda example_proto, feature1, feature2: _parse_pair(
    example_proto, feature1, feature2, fp_dim=(56, 56, 6)
)
_parse_pair_112 = lambda example_proto, feature1, feature2: _parse_pair(
    example_proto, feature1, feature2, fp_dim=(112, 112, 6)
)


_parse_single_56 = lambda example_proto, feature_id: _parse_single(
    example_proto, feature_id, fp_dim=(56, 56, 6)
)
_parse_single_56 = lambda example_proto, feature_id: _parse_single(
    example_proto, feature_id, fp_dim=(112, 112, 6)
)

### conversion funtions


def _onehot_year(fp, year):
    year_onehot = tf.one_hot(year, 10)  # 1970~4 -> 0, 2015~9 -> 9
    return (fp, year_onehot)


### dataset functions


def create_dataset(filepath, map_functions=[]):
    dataset = tf.data.TFRecordDataset(filepath, compression_type="GZIP")

    ### preprocess the features
    for func in map_functions:
        dataset = dataset.map(func, num_parallel_calls=4)

    return dataset


def create_pair_56_dataset(
    filepath, feature1="floorplan", feature2="year", map_functions=[]
):
    return create_dataset(
        filepath,
        [lambda example_proto: _parse_pair_56(example_proto, feature1, feature2)]
        + map_functions,
    )


def create_single_dataset(filepath, feature_id="year", map_functions=[]):
    return create_dataset(
        filepath,
        [lambda example_proto: _parse_single(example_proto, feature_id)]
        + map_functions,
    )


### VGG 5y model


def VGG16_convolutions(fp_dim=(56, 56, 6)):
    if K.image_data_format() == "channels_last":
        input_shape = (fp_dim[0], fp_dim[1], fp_dim[2])
    else:
        input_shape = (fp_dim[2], fp_dim[0], fp_dim[1])

    model = Sequential()
    model.add(GaussianNoise(0.1, input_shape=input_shape))

    model.add(Conv2D(64, (3, 3), activation="relu", name="conv1_1", padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", name="conv1_2", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(1, 1), padding="same"))

    model.add(Conv2D(128, (3, 3), activation="relu", name="conv2_1", padding="same"))
    model.add(Conv2D(128, (3, 3), activation="relu", name="conv2_2", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(1, 1), padding="same"))

    model.add(Conv2D(256, (3, 3), activation="relu", name="conv3_1", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", name="conv3_2", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", name="conv3_3", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation="relu", name="conv4_1", padding="same"))
    model.add(Conv2D(512, (3, 3), activation="relu", name="conv4_2", padding="same"))
    model.add(Conv2D(512, (3, 3), activation="relu", name="conv4_3", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation="relu", name="conv5_1", padding="same"))
    model.add(Conv2D(512, (3, 3), activation="relu", name="conv5_2", padding="same"))
    model.add(Conv2D(512, (3, 3), activation="relu", name="conv5_3", padding="same"))
    return model


def create_vgg_5y_model():
    model = VGG16_convolutions()

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


### standalone run
if __name__ == "__main__":
    print("Tensorflow version: ", tf.version.VERSION)  # tf2
    print("Keras version: ", tf.keras.__version__)  # 2.2.4-tf

    print("Is eager execution enabled: ", tf.executing_eagerly())
    print(tf.config.list_physical_devices("GPU"))  # tf2

    model = create_vgg_5y_model()
    print("Done")
