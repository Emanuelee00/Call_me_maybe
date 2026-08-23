import json
from typing import Type

from pydantic import BaseModel, ValidationError


def load_and_validate_definitions(path: str, cls: Type[BaseModel]) -> list[BaseModel]:
    """Load a JSON array from path and validate each item against cls."""
    try:
        with open(path, "r") as f:
            items = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Input file not found: {path}")
    except json.JSONDecodeError as error:
        raise ValueError(f"Input file {path} is not valid JSON: {error}")

    try:
        return [cls.model_validate(item) for item in items]
    except ValidationError as error:
        raise ValueError(f"Input file {path} does not match the expected schema: {error}")
