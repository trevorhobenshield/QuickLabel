import shutil
from pathlib import Path


def undo_labeling(img_directory: str) -> None:
    """
    Undo labeling performed by `label.py`

    :param img_directory: path to image directory that previously had it's contents labeled
    :return: None
    """
    # move images out of directories
    for d in [f for f in Path(img_directory).iterdir() if f.is_dir()]:
        for img in d.iterdir():
            new = d.parent / img.name
            print('*', img, '->', new)
            Path(img).rename(new)
    # remove directories
    [shutil.rmtree(p) for p in Path(img_directory).iterdir() if p.is_dir()]


def main():
    undo_labeling('images')


if __name__ == '__main__':
    main()

