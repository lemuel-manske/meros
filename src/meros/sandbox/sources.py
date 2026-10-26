from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    individual_id: int
    path: str


# TODO: replace by project.metadata
sources = [
    Source(individual_id=101, path="data/media/masked_crops/101_11/0/250.png"),
    # Source(individual_id=101, path="data/media/masked_crops/101_12/0/13.png"),
    Source(individual_id=65, path="data/media/masked_crops/65_7/0/179.png"),
    # Source(individual_id=65, path="data/media/masked_crops/65_7/0/210.png"),
    Source(individual_id=11, path="data/media/masked_crops/h01/0/340.png"),
    Source(individual_id=12, path="data/media/masked_crops/h02/0/1170.png"),
    Source(individual_id=13, path="data/media/masked_crops/h03/0/3353.png"),
    Source(individual_id=14, path="data/media/masked_crops/h04/0/2877.png"),
]
