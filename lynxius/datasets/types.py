from dataclasses import dataclass
from typing import List


@dataclass
class Dataset:
    uuid: str
    date_created: str
    organization_uuid: str
    organization_name: str


@dataclass
class DatasetEntry:
    uuid: str
    dataset_uuid: str
    query: str
    output: str
    reference: str
    score: str
    comments: str
    date_created: str
    date_modified: str


class DatasetDetails:
    dataset: Dataset
    entries: List[DatasetEntry]
