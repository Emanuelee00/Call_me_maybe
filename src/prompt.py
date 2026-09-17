from typing import TYPE_CHECKING

from .models import FunctionDefinition
from .vocab import encode_to_ids

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


def format_functions_list(
    functions_list: list[tuple[str, list[str], str, str]],
) -> str:
    """Render the functions from build_system_prompt as a plain-text list.

    Args:
        functions_list: Tuples of (name, params, description, return
            type), as produced by build_system_prompt.

    Returns:
        A human-readable "Available functions:" block, one line per
        function.
    """
    lines = ["Available functions:"]
    for name, params, description, return_type in functions_list:
        lines.append(f"- {name}({', '.join(params)}): {description}")
    return "\n".join(lines)


def build_system_prompt(
    functions: list[FunctionDefinition],
) -> list[tuple[str, list[str], str, str]]:
    """Turn functions into (name, params, description, return type) tuples.

    Args:
        functions: The parsed function definitions.

    Returns:
        One tuple per function: its name, a list of "param: type"
        strings, its description, and its return type.
    """
    output = []
    for function in functions:
        output.append(
            (
                function.name,
                [
                    f"{name}: {schema.type}"
                    for name, schema in function.parameters.items()
                ],
                function.description,
                function.returns.type,
            )
        )
    return output


def build_full_prompt(formatted_system_prompt: str, user_prompt: str) -> str:
    """Append the user prompt and the opening JSON to the system prompt.

    Args:
        formatted_system_prompt: The "Available functions:" block.
        user_prompt: The natural-language question to answer.

    Returns:
        The full prompt text the model will continue from, primed
        with the opening of the expected JSON.
    """
    full_prompt = (
        formatted_system_prompt + f'\nUser: {user_prompt} \n JSON: {{"name": "'
    )
    return full_prompt


def add_parameters_to_ids(
    model: "Small_LLM_Model", current_ids: list[int]
) -> list[int]:
    """Open the "parameters" object in the JSON being built.

    Args:
        model: The LLM wrapper, used to encode the literal text.
        current_ids: Token ids generated so far.

    Returns:
        current_ids with the primed ', "parameters" : {' text
        appended.
    """
    new_ids = encode_to_ids(model, '", "parameters" : {')
    return current_ids + new_ids
