from models import FunctionDefinition
from main import load_and_validate_definitions, parse_arguments
import json
from llm_sdk import Small_LLM_Model
import re

def load_vocab(path):
    with open(path) as f:
        token_to_id = json.load(f)
    id_to_token = {v: k for k, v in token_to_id.items()}
        
    return token_to_id, id_to_token



def format_output(string: str) -> str:
    lines = ["Available functions:"]
    for name, params, description, return_type in string:
        lines.append(f"- {name}({', '.join(params)}): {description}")
    return "\n".join(lines)


def build_system_prompt(functions: list[FunctionDefinition]) -> list:
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


def build_full_prompt(formatted_system_prompt: str, user_prompt: str):
    full_prompt = (
        formatted_system_prompt + f'\nUser: {user_prompt} \n JSON: {{"name": "'
    )
    return full_prompt

def get_fn_name_tokens(token_to_id: dict) -> dict:
    result = {}
    for token, tid in token_to_id.items():
        if re.match(r'^[a-zA-Z0-9_]+$', token):
            result[tid] = token
    return result


def generate_fn_name(
    model,
    input_ids: list[int],
    valid_names: list[str],
    fn_tokens: dict[int, str],
) -> str:
    prefix = ""
    current_ids = list(input_ids)

    while prefix not in valid_names:
        logits = model.get_logits_from_input_ids(current_ids)

        best_id = None
        best_logit = float("-inf")

        for tid, token in fn_tokens.items():
            candidate = prefix + token
            for name in valid_names:
                if name.startswith(candidate):
                    if logits[tid] > best_logit:
                        best_logit = logits[tid]
                        best_id = tid
                    break

        if best_id is None:
            raise ValueError(f"No valid token found at prefix: {prefix!r}")

        prefix += fn_tokens[best_id]
        current_ids.append(best_id)

    return prefix



if __name__ == "__main__":
    args = parse_arguments()
    functions = load_and_validate_definitions(
        args.functions_definition, FunctionDefinition
    )
    functions_list = build_system_prompt(functions)
    formatted_functions = format_output(functions_list)
    user_prompt = "Calculate the square root of 3"
    full_formatted_prompt = build_full_prompt(formatted_functions, user_prompt)
    print(build_full_prompt(formatted_functions, user_prompt))

    model = Small_LLM_Model()
    tensor = model.encode(full_formatted_prompt)
    input_ids = [int(x) for x in tensor[0]]
    path = model.get_path_to_vocab_file()
    token_to_id, id_to_token = load_vocab(path)
    print(dict(list(token_to_id.items())[:5]))
    print(dict(list(id_to_token.items())[:5]))
    #print(input_ids[:5])
    fn_tokens = get_fn_name_tokens(token_to_id)
    print(f"Token validi per nomi funzione: {len(fn_tokens)}")
    print(dict(list(fn_tokens.items())[:100]))
    print()
    fn_tokens = get_fn_name_tokens(token_to_id)
    fn_name = generate_fn_name(model, input_ids, [f.name for f in functions], fn_tokens)
    print("Funzione scelta:", fn_name)


