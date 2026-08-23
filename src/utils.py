import json
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def load_and_validate_definitions(path: str, cls: Type[T]) -> list[T]:
    """Load a JSON array from path and validate each item against cls.

    Args:
        path: Path to a JSON file holding an array of objects.
        cls: The pydantic model each array item must match.

    Returns:
        One validated cls instance per array item.

    Raises:
        ValueError: If the file is missing, not valid JSON, or an
            item doesn't match cls's schema.
    """
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
        raise ValueError(
            f"Input file {path} does not match the expected schema: {error}"
        )
