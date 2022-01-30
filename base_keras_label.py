import re
import shutil
from glob import glob
from pathlib import Path
from typing import Optional, Any, Generator

import PIL
from keras import Model
from keras.applications.nasnet import NASNetMobile, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from numpy import r_
from tqdm import tqdm


def imgs_preds(model_config: tuple[Model, tuple[int, int]], path: str, slc: Optional[slice] = None) -> list[
    list[str, str]]:
    """
    Get the predicted class for each image

    :param model_config: model object and image dimensions that the model expects to receive
    :param path: path to image directory
    :param slc: slice of images from directory
    :return: list of lists containing image filenames and their predicted classes
    """
    images, res = glob(f'{path}/*')[slc] if slc else glob(f'{path}/*'), []
    for img in tqdm((lambda x, y: filter(re.compile(x).match, y))(r'.*\.(jpg|png|jpeg)', images), total=len(images)):
        try:
            image = preprocess_input(r_[[img_to_array(load_img(img, target_size=model_config[1]))]])
            pred = model_config[0].predict(image)
            res.append([Path(img).name, decode_predictions(pred)[0][0][1]])
        except (OSError, FileNotFoundError, PIL.UnidentifiedImageError) as e:
            print('\n', e)
    return res


def find_matching(preds: list[list, ...], keywords: list[str]) -> Generator[tuple[str, str], Any, None]:
    """
    Return only images whose class predictions match the keywords

    :param keywords: list of classes you hope to find in the images
    :param preds: generator of tuples containing image filenames and their predicted classes
    :return: generator of tuples containing image filenames and their predicted classes
    """
    return ((i, c) for i, c in preds for k in keywords if c == k)


def label(path: str, matches: Generator[tuple[str, str], Any, None]) -> None:
    """
    Perform the labeling operation

    Create directories for predicted classes that do not exist yet,
    and add images to the existing respective class directories

    :param path: path to image directory that previously had its contents labeled
    :param matches: generator of tuples containing image filenames and their predicted classes
    """
    matches = list(matches) # need to convert generator here
    [Path(f'{path}/{cls}').mkdir(parents=True, exist_ok=False) for cls in
     {c for _, c in matches} - {f.name for f in Path(path).iterdir() if f.is_dir()}]
    [print(f"{(o := f'{path}/{img}')} -> {Path(o).rename(f'{path}/{cls}/{img}')}") for img, cls in matches]


def undo_labeling(path: str) -> None:
    """
    Undo labeling performed by `label()`

    :param path: path to image directory that previously had its contents labeled
    :return: None
    """
    [print(f'* {y} -> {Path(y).rename((x.parent / y.name))}') for x in
     (z for z in Path(path).iterdir() if z.is_dir()) for y in x.iterdir()]
    [shutil.rmtree(_) for _ in Path(path).iterdir() if _.is_dir()]


def run(model_config: tuple[Model, tuple[int, int]], path: str = '', slc: Optional[slice] = None,
        keywords: list[str] = None) -> None:
    """
    Run the program

    :param model_config: model object and image dimensions that the model expects to receive
    :param path: path to image directory
    :param slc: slice of images from directory
    :param keywords: list of classes you hope to find in the images
    """
    label(path, find_matching(imgs_preds(model_config, path, slc), keywords))


def main():
    run(
        model_config=(NASNetMobile(), (224, 224)),  # fast and accurate
        path='images/',
        # slc=slice(0, 5),  # define a small slice of images to operate on for testing
        keywords=Path('keywords.txt').read_text().splitlines()
    )

    # undo_labeling(img_dir)


if __name__ == '__main__':
    main()
