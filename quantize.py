import os
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TFHUB_CACHE_DIR'] = '/home/x/tfhub_modules'  # todo: change to user cache dir

import re
from glob import glob
from typing import Optional
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

[tf.config.experimental.set_memory_growth(d, 1) for d in tf.config.list_physical_devices('GPU')]


def quantize(model_url: str,
             input_shape: tuple[int, ...],
             quantization: str,
             path: Optional[str] = '',
             representative_dataset: Optional[None] = None) -> str:
    """
    Perform PTQ on model
    :param model_url: url of the model on tfhub as a string
    :param input_shape: shape of images model accepts (3 channel image)
    :param quantization: type of quantization
    :param path: path to save tflite model
    :param representative_dataset: dataset to include
    :return: path to quantized model as a string
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        hub.KerasLayer(model_url)
    ]))
    quantization = quantization.strip().lower()

    # Default (dynamic)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Full integer quantization with float fallback (default float I/O)
    # Need representative_dataset
    if quantization == 'fallback':
        converter.representative_dataset = representative_dataset

    # Full integer quantization (unsigned 8-bit, integer only)
    # Need representative_dataset
    elif quantization == 'uint8':
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    # Full integer quantization (signed 8-bit, integer only)
    # Need representative_dataset
    elif quantization == 'int8':
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    # Float16 quantization
    elif quantization == 'float16':
        converter.target_spec.supported_types = [tf.float16]

    # Integer only: 16-bit activations with 8-bit weights
    # Need representative_dataset
    elif quantization == '16x8':
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
    if not path:
        fname = f"{model_url.split('tfhub.dev/')[-1].replace('/', '_')}_qt_{quantization}"
        x = 'rishit-dagli sayakpaul ml-kit google tensorflow agripredict adityakane2001 imagenet'.split()
        for u in x:
            fname = fname.replace(u, '')
        tflite_buffer = converter.convert()
        tf.io.gfile.GFile(f'models/{fname.lstrip("_")}.tflite', 'wb').write(tflite_buffer)
    else:
        fname = f'{path}.tflite'
        tflite_buffer = converter.convert()
        tf.io.gfile.GFile(fname, 'wb').write(tflite_buffer)
    return fname


def main():
    """
    Quantize model

    Note: The dimensions in `(Image.open(image_path).resize((x, y)))`
          should match the dimensions specified in `input_shape`

    :return:
    """

    def representative_dataset():
        """
        Representative dataset generator

        For full integer quantization (tf.int8, tf.uint8), you need to calibrate the range
        of all floating-point tensors in the model by including a small subset of
        the training or validation data
        :return:
        """
        representative_images = list(
            (lambda p, s: filter(re.compile(p).match, s))(r'.*\.(jpg|png|jpeg)', glob('representative_dataset/*')))
        print('# images = ', len(representative_images))
        for image_path in representative_images:
            yield [np.r_[[np.array(Image.open(image_path).resize((224, 224))) / 255.]].astype(np.float32)]

    quantize(**{
        'model_url': 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/classification/2',
        'input_shape': (384, 384, 3),  # match representative_dataset data shape with this shape
        'quantization': 'float16',  # 16x8 and float16 currently supported
        'representative_dataset': representative_dataset
    })


if __name__ == '__main__':
    main()
