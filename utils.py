import json


def load_and_validate_definitions(path: str, cls: Type[BaseModel]) -> list[BaseModel]:
    with open(path, "r") as f:
        functions = json.load(f)
    functions = [cls.model_validate(item) for (item) in functions]
    return functions
