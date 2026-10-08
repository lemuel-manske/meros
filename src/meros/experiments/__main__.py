"""Run any experiment module through the same command."""

import argparse
import importlib
import re
import sys

from typing import Protocol, cast


class ExperimentModule(Protocol):
    def main(self, argv: list[str] | None = None) -> None: ...


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("experiment", help="Experiment identifier, for example 002 or pose")

    arguments = sys.argv[1:] if argv is None else argv

    args = parser.parse_args(arguments[:1])

    remaining = arguments[1:]

    if not re.fullmatch(r"[A-Za-z0-9_]+", args.experiment):
        parser.error("Use letters, digits, or underscores in the experiment identifier")

    name = f"meros.experiments.experiment_{args.experiment}"

    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise

        parser.error(f"Experiment {args.experiment!r} does not exist")

    experiment = cast(ExperimentModule, module)

    experiment.main(remaining)


if __name__ == "__main__":
    main()
