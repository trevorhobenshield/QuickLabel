import os
import re
import warnings
from time import time
from typing import Optional

import PIL
import joblib

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TFHUB_CACHE_DIR'] = '/home/x/tfhub_modules'  # keep everything in here instead of in /tmp

from numpy import r_, array
import tensorflow as tf
from PIL import Image
import numpy as np
from glob import glob
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed

[tf.config.experimental.set_memory_growth(d, 1) for d in tf.config.list_physical_devices('GPU')]


def label_guess(cache: str,
                model: tuple[tuple[str, int]],
                image_path: str) -> list:
    """
    Crude guess function so that you don't have to input a label filename for each model.
    :param cache: tfhub_modules cache dir
    :param model: tuple of (model_name, input_shape)
    :param input_size: image dim that model expects
    :param image_path: path to input image
    :return: list of class labels
    """
    img = r_[[array(Image.open(glob(f'{image_path}/*')[0]).resize(model[1])) / 255.]].astype(np.float32)
    intp = tf.lite.Interpreter(model_path=f'{cache}/{model[0]}')
    intp.allocate_tensors()
    intp.set_tensor(intp.get_input_details()[0]['index'], img)
    intp.invoke()
    tensor = intp.get_tensor(intp.get_output_details()[0]['index'])
    cls = ...
    if tensor.shape[1] == 1000:
        cls = Path('labels/ilsvrc2012_wordnet_lemmas.txt').read_text().splitlines()
    elif tensor.shape[1] == 1001:
        cls = Path('labels/ImageNetLabels.txt').read_text().splitlines()
    elif tensor.shape[1] == 21_843:
        cls = Path('labels/imagenet21k_wordnet_lemmas.txt').read_text().splitlines()
    return cls


def infer(cache: str,
          model: tuple[str, tuple[int, ...]],
          img_dir: str) -> dict:
    """
    Run inference on all images with the specified model
    :param cache: path to cache dir holding models
    :param model: tuple of (model_name, input_shape)
    :param img_dir: path to image dir
    :return: dict of model name and predictions
    """

    labels = label_guess(cache, model, img_dir)
    images = glob(f'{img_dir}/*')
    num_images = len(images)
    start = time()
    img_preds = []
    for i, image in enumerate((lambda x, y: filter(re.compile(x).match, y))(r'.*\.(jpg|png|jpeg)', images)):
        if not i % 1_000:
            print(f'\n[processed {i}/{num_images} images]\t[{(time() - start) / 60:.2f} mins elapsed]\n')
        try:
            intp = tf.lite.Interpreter(model_path=f'{cache}/{model[0]}')
            intp.allocate_tensors()
            intp.set_tensor(intp.get_input_details()[0]['index'],
                            r_[[array(Image.open(image).resize(model[1])) / 255.]].astype(np.float32))
            intp.invoke()
            tensor = intp.get_tensor(intp.get_output_details()[0]['index'])
            pred = labels[tf.math.argmax(tensor, axis=-1)[0]]
            print(f'{Path(image).name} ({pred})')
            img_preds.append([Path(image).name, pred])
        except (OSError, FileNotFoundError, PIL.UnidentifiedImageError) as e:
            print('\n', e)
    return dict(img_preds)


def run_models_parallel(cache: str,
                        models: list[tuple[str, tuple[int, ...]]],
                        img_dir: str,
                        keywords: str) -> None:
    """
    If you don't care about visualizations, you can just run the program.
    :param cache: path to cache dir holding models
    :param keywords: path to keywords .txt file
    :param img_dir: path to image dir
    :param models: list of tuples (model_name, input_shape) with models to run.
    :return: None
    """
    df = pd.DataFrame(Parallel(n_jobs=-1)(delayed(infer)(cache, m, img_dir) for m in models))
    img_cls = [[c, k] for k in Path(keywords).read_text().splitlines() for c in df.columns if
               k in df[c].to_numpy()]
    [Path(f'{img_dir}/{cls}').mkdir(parents=True, exist_ok=False) for cls in
     set(c for _, c in img_cls) - {f.name for f in Path(f'{img_dir}').iterdir() if f.is_dir()}]
    [print(f"{(o := f'{img_dir}/{img}')} ->", Path(o).rename(f'{img_dir}/{cls}/{img}')) for img, cls in img_cls]


def image_batches(cache: str,
                  model: tuple[str, tuple[int, ...]],
                  images: list,
                  labels: list) -> dict:
    """
    Run a single model over batches of images in parallel
    :param cache: path to cache dir holding models
    :param model: tuple of (model_name, input_shape)
    :param images: path to image dir
    :param labels: path to labels file
    :return: dict of model name and predictions
    """
    img_preds = []
    for image in list((lambda x, y: filter(re.compile(x).match, y))(r'.*\.(jpg|png|jpeg)', images)):
        try:
            intp = tf.lite.Interpreter(model_path=f'{cache}/{model[0]}')
            intp.allocate_tensors()
            intp.set_tensor(intp.get_input_details()[0]['index'],
                            r_[[np.array(Image.open(image).resize(model[1])) / 255.]].astype(np.float32))
            intp.invoke()
            tensor = intp.get_tensor(intp.get_output_details()[0]['index'])
            pred = labels[tf.math.argmax(tensor, axis=-1)[0]]
            print(f'{Path(image).name} ({pred})')
            img_preds.append([Path(image).name, pred])
        except (OSError, FileNotFoundError, PIL.UnidentifiedImageError) as e:
            print('\n', e)
    return dict(img_preds)


def run_image_batches(cache: str,
                      model: tuple[str, tuple[int, ...]],
                      img_dir: str,
                      keywords: str,
                      n_jobs=-1,
                      threads: Optional[Optional[str]] = None,
                      n=joblib.cpu_count()) -> None:
    """
    Call `parallel_img_batches()` to run model over batches of images in parallel
    :param cache: path to cache dir holding models
    :param model: tuple of (model_name, input_shape)
    :param img_dir: path to image dir
    :param keywords: path to keywords .txt file
    :param n_jobs: joblib param
    :param threads: joblib param
    :param n: number of cores available
    :return: None
    """
    images = glob(fr'{img_dir}\*')
    labels = label_guess(cache, model, img_dir)
    df = pd.DataFrame(Parallel(n_jobs=n_jobs, prefer=threads)(
        delayed(image_batches)(cache, model, img, labels) for img in np.array_split(r_[images], n)))
    img_cls = [[c, k] for k in Path(keywords).read_text().splitlines() for c in df.columns if
               k in df[c].to_numpy()]
    [Path(f'{img_dir}/{cls}').mkdir(parents=True, exist_ok=False) for cls in
     set(c for _, c in img_cls) - {f.name for f in Path(f'{img_dir}').iterdir() if f.is_dir()}]
    [print(f"{(o := f'{img_dir}/{img}')} ->", Path(o).rename(f'{img_dir}/{cls}/{img}')) for img, cls in img_cls]


def sequential(cache: str,
               models: list[tuple[str, tuple[int, ...]]],
               img_dir: str,
               keywords: str) -> None:
    """
    Run models sequentially
    :param cache: path to cache dir holding models
    :param keywords: path to keywords .txt file
    :param img_dir: path to image dir
    :param models: list of tuples (model_name, input_shape) with models to run.
    :return: None
    """
    preds = []
    for model in models:
        preds.append(infer(cache, model, img_dir))
    df = pd.DataFrame(preds)

    keywords = Path(keywords).read_text().splitlines()

    images_and_classes = [[col, k] for k in keywords for col in df.columns if k in df[col].to_numpy()]
    print('found classes:', {x[1] for x in images_and_classes})
    # df.T.rename({i: name for i, name in enumerate(models_)}, axis=1)

    existing_classes = {f.name for f in Path(img_dir).iterdir() if f.is_dir()}
    matched_classes = set(cls for _, cls in images_and_classes)
    classes = matched_classes - existing_classes

    print(f'creating new class dirs: {classes}')
    for cls in classes:
        Path(fr'{img_dir}\{cls}').mkdir(parents=True, exist_ok=False)

    for img, cls in images_and_classes:
        try:
            o = fr'{img_dir}\{img}'
            n = fr'{img_dir}\{cls}\{img}'
            print(f'{o} -> {n}')
            Path(o).rename(n)
        except Exception as e:
            print(e)


def main():
    ## Run multiple models in parallel
    run_models_parallel(cache='models',
             models=[
                 ('nasnet_mobile_classification_5_qt_float16.tflite', (224, 224)),
                 ('nasnet_mobile_classification_5_qt_16x8.tflite', (224, 224)),
             ],
             img_dir='images',
             keywords='keywords.txt')

    ## Run a copies of a single model over batches of images in parallel
    # run_image_batches(cache='models',
    #                   model=('nasnet_mobile_classification_5_qt_float16.tflite', (224, 224)),
    #                   img_dir='images',
    #                   keywords='keywords.txt')

    ## Slowly run each model sequentially
    # sequential(cache='models',
    #            models=[
    #                ('nasnet_mobile_classification_5_qt_float16.tflite', (224, 224)),
    #                ('nasnet_mobile_classification_5_qt_16x8.tflite', (224, 224)),
    #            ],
    #            img_dir='images',
    #            keywords='keywords.txt')


if __name__ == '__main__':
    main()
