import argparse


def positive_int(value: str) -> int:
    """
    Convert a string to a positive integer.
    """
    number = int(value)

    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")

    return number
