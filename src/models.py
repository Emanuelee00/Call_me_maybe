from pydantic import BaseModel
from typing import Any


class ParameterSchema(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ParameterSchema


class Prompt(BaseModel):
    prompt: str


class FunctionCallRequest(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
