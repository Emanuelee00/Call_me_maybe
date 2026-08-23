import json
from typing import Type

from pydantic import BaseModel


def load_and_validate_definitions(path: str, cls: Type[BaseModel]) -> list[BaseModel]:
    with open(path, "r") as f:
        items = json.load(f)
    return [cls.model_validate(item) for item in items]
