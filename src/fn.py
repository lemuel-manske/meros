import argparse

from pathlib import Path


def glob_jpgs(path: str) -> list[Path]:
    """
    Glob all .jpg files in the given path.
    """
    return list(Path(path).glob("*.jpg"))


def glob_csvs(path: str) -> list[Path]:
    """
    Glob all .csv files in the given path.
    """
    return list(Path(path).glob("*.csv"))


def positive_int(value: str) -> int:
    """
    Convert a string to a positive integer.
    """
    number = int(value)

    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")

    return number
