from pydantic import BaseModel
from typing import Any


class ParameterSchema(BaseModel):
    """The declared type of a function parameter or return value.

    Attributes:
        type: One of "number", "integer", "string", "boolean".
    """

    type: str


class FunctionDefinition(BaseModel):
    """One callable function, as declared in functions_definition.json.

    Attributes:
        name: The function's exact name.
        description: Human-readable description, shown to the model.
        parameters: Each parameter's name mapped to its schema.
        returns: The function's return type.
    """

    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ParameterSchema


class Prompt(BaseModel):
    """One natural-language request from function_calling_tests.json.

    Attributes:
        prompt: The request text.
    """

    prompt: str


class FunctionCallRequest(BaseModel):
    """One resolved function call, as written to the output file.

    Attributes:
        prompt: The original natural-language request.
        name: The chosen function's name.
        parameters: The generated arguments, by parameter name.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]
