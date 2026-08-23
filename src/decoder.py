import json
import re
from typing import TYPE_CHECKING

from .models import FunctionDefinition, ParameterSchema

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


def load_vocab(path: str) -> tuple[dict[str, int], dict[int, str]]:
    """Load the tokenizer vocab file and return (token -> id, id -> token)."""
    with open(path) as f:
        token_to_id = json.load(f)
    id_to_token = {v: k for k, v in token_to_id.items()}

    return token_to_id, id_to_token


def format_output(functions_list: list[tuple[str, list[str], str, str]]) -> str:
    """Render the functions built by build_system_prompt as a plain-text list."""
    lines = ["Available functions:"]
    for name, params, description, return_type in functions_list:
        lines.append(f"- {name}({', '.join(params)}): {description}")
    return "\n".join(lines)


def build_system_prompt(functions: list[FunctionDefinition]) -> list[tuple[str, list[str], str, str]]:
    """Turn function definitions into (name, param descriptions, description, return type) tuples."""
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
    """Append the user prompt and the opening of the expected JSON to the system prompt."""
    full_prompt = (
        formatted_system_prompt + f'\nUser: {user_prompt} \n JSON: {{"name": "'
    )
    return full_prompt


def get_filtered_tokens_vocab(token_to_id: dict[str, int], regex: str | None = None) -> dict[int, str]:
    """Keep only the vocab tokens whose whole text matches the given character class."""
    result = {}
    for token, token_id in token_to_id.items():
        if re.match(f'^[{regex}]+$', token):
            result[token_id] = token
    return result


def constrain_functions(
    model: "Small_LLM_Model",
    encoded_token: list[int],
    valid_names: list[str],
    valid_tokens_filter_dict: dict[int, str],
) -> tuple[str, list[int]]:
    """Generate token by token until the result exactly matches one of valid_names."""
    prefix = ""
    current_ids = list(encoded_token)

    while prefix not in valid_names:
        logits = model.get_logits_from_input_ids(current_ids)

        candidate_ids = set()
        for token_id, token in valid_tokens_filter_dict.items():
            candidate = prefix + token
            for name in valid_names:
                if name.startswith(candidate):
                    candidate_ids.add(token_id)
                    break

        if not candidate_ids:
            raise ValueError(f"No valid token found at prefix: {prefix!r}")

        best_token_id = masked_argmax(logits, candidate_ids)
        prefix += valid_tokens_filter_dict[best_token_id]
        current_ids.append(best_token_id)

    return prefix, current_ids


def masked_argmax(logits: list[float], candidate_ids: set[int]) -> int:
    """Pick the id with the highest logit, considering only candidate_ids."""
    return max(candidate_ids, key=lambda token_id: logits[token_id])


def encode_to_ids(model: "Small_LLM_Model", text: str) -> list[int]:
    """Encode text and return it as a plain list of int token ids."""
    tensor = model.encode(text)
    ids = [int(x) for x in tensor[0]]
    return ids


def constrain_number_parameters(
    model: "Small_LLM_Model",
    encoded_tokens: list[int],
    valid_tokens_filter_dict: dict[int, str],
    minus_token_id: int,
    stop_token_id: int,
    max_tokens: int = 200,
) -> list[int]:
    """Generate a number or free-text string value, masked to valid chars, until the stop token wins."""
    generated: list[int] = []
    stopped = False

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(encoded_tokens + generated)

        candidate_ids = set(valid_tokens_filter_dict)
        if not generated:
            candidate_ids.add(minus_token_id)
        if generated:
            candidate_ids.add(stop_token_id)

        best_token_id = masked_argmax(logits, candidate_ids)
        if best_token_id == stop_token_id:
            stopped = True
            break

        generated.append(best_token_id)

    if not stopped:
        raise ValueError("Number too long: exceeded max_tokens without hitting the stop token")

    return generated


def get_regex_for_type(parameter: ParameterSchema) -> str:
    """Character class allowed for this parameter's value (number chars, or anything but a quote)."""
    TYPE_REGEX = {
        "number": "0-9.",
        "string": '^"',
    }

    return TYPE_REGEX[parameter.type]


def generate_parameters(
    model: "Small_LLM_Model",
    encoded_tokens: list[int],
    parameters: dict[str, ParameterSchema],
    token_to_id: dict[str, int],
) -> list[int]:
    """Append every parameter's key and constrained value to encoded_tokens, then close the object."""
    encoded_comma_and_space = encode_to_ids(model, ', ')
    encoded_end = encode_to_ids(model, '}}')
    encoded_quotes = encode_to_ids(model, '"')
    encoded_minus = encode_to_ids(model, '-')
    for i, (key, parameter) in enumerate(parameters.items()):
        encoded_key = encode_to_ids(model, f'"{key}": ')
        encoded_tokens.extend(encoded_key)

        if parameter.type == "boolean":
            boolean_vocab = get_filtered_tokens_vocab(token_to_id, "a-zA-Z")
            _, encoded_tokens = constrain_functions(model, encoded_tokens, ["true", "false"], boolean_vocab)
        else:
            if parameter.type == "string":
                encoded_tokens.extend(encoded_quotes)
            valid_tokens_filter = get_regex_for_type(parameter)
            valid_tokens_filter_vocab = get_filtered_tokens_vocab(token_to_id, valid_tokens_filter)
            if parameter.type == "string":
                stop_token_id = encoded_quotes[0]
            elif i < len(parameters) - 1:
                stop_token_id = encoded_comma_and_space[0]
            else:
                stop_token_id = encoded_end[0]
            current_tokens = constrain_number_parameters(model, encoded_tokens, valid_tokens_filter_vocab, encoded_minus[0], stop_token_id, max_tokens=200)
            encoded_tokens.extend(current_tokens)
            if parameter.type == "string":
                encoded_tokens.extend(encoded_quotes)

        if i < len(parameters) - 1:
            encoded_tokens.extend(encoded_comma_and_space)
    encoded_tokens.extend(encoded_end)
    return encoded_tokens


def add_parameters_to_ids(model: "Small_LLM_Model", current_ids: list[int]) -> list[int]:
    """Open the "parameters" object in the JSON being built."""
    new_ids = encode_to_ids(model, '", "parameters" : {')
    return current_ids + new_ids


def process_prompt(
    model: "Small_LLM_Model",
    token_to_id: dict[str, int],
    functions: list[FunctionDefinition],
    formatted_functions: str,
    valid_tokens_for_name_func: dict[int, str],
    prompt_text: str,
) -> dict[str, object]:
    """Run the full pipeline for one prompt and return {prompt, name, parameters}."""
    full_formatted_prompt = build_full_prompt(formatted_functions, prompt_text)
    encoded_token_prompt_ids = encode_to_ids(model, full_formatted_prompt)

    fn_name, current_ids = constrain_functions(
        model, encoded_token_prompt_ids, [f.name for f in functions], valid_tokens_for_name_func
    )
    parameters = next(f.parameters for f in functions if f.name == fn_name)

    current_ids = add_parameters_to_ids(model, current_ids)
    current_ids = generate_parameters(model, current_ids, parameters, token_to_id)

    decoded_text = model.decode(current_ids)
    json_start = decoded_text.index('{"name"')
    parsed = json.loads(decoded_text[json_start:])

    return {
        "prompt": prompt_text,
        "name": parsed["name"],
        "parameters": parsed["parameters"],
    }
