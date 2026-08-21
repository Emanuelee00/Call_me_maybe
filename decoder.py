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
    encoded_token_prompt_ids: list[int],
    valid_names: list[str],
    valid_tokens_for_name_func: dict[int, str],
) -> str:
    prefix = ""
    current_ids = list(encoded_token_prompt_ids)

    while prefix not in valid_names:
        print(prefix)
        logits = model.get_logits_from_input_ids(encoded_token_prompt_ids)

        best_id = None
        best_logit = float("-inf")
        oken_to_id, id_to_token = load_vocab(path)

        for i, (tid, token) in enumerate(valid_tokens_for_name_func.items()):
            candidate = prefix + token
            print("candidate",candidate) if i%1000 == 0 else None
            for name in valid_names:
                if name.startswith(candidate):
                    if logits[tid] > best_logit:
                        print("*"*50)
                        print("tid", tid, "logits[tid]",logits[tid], "value:", id_to_token[tid])
                        best_logit = logits[tid]
                        best_id = tid
                    break

        if best_id is None:
            raise ValueError(f"No valid token found at prefix: {prefix!r}")

        prefix += valid_tokens_for_name_func[best_id]
        current_ids.append(best_id)

    return prefix, current_ids

def add_parameters_to_ids(model,  current_ids: list[int]):
    output = f'", "parameters": {{'
    tensor = model.encode(output)
    new_ids = [int(x) for x in tensor[0]]
    return current_ids + new_ids



if __name__ == "__main__":
    args = parse_arguments()
    functions = load_and_validate_definitions(
        args.functions_definition, FunctionDefinition
    )
    functions_list = build_system_prompt(functions)
    formatted_functions = format_output(functions_list)
    user_prompt = " caluclate the root of 3"
    full_formatted_prompt = build_full_prompt(formatted_functions, user_prompt)
    print(build_full_prompt(formatted_functions, user_prompt))

    model = Small_LLM_Model()
    tensor = model.encode(full_formatted_prompt)
    encoded_token_prompt_ids = [int(x) for x in tensor[0]]
    path = model.get_path_to_vocab_file()
    token_to_id, id_to_token = load_vocab(path)
    print(dict(list(token_to_id.items())[:5]))
    print(dict(list(id_to_token.items())[:5]))
    #print(encoded_token_prompt_ids[:5])
    valid_tokens_for_name_func = get_fn_name_tokens(token_to_id)
    print(f"Token validi per nomi funzione: {len(valid_tokens_for_name_func)}")
    print(dict(list(valid_tokens_for_name_func.items())[:100]))
    print()
    """ 
        for ids in encoded_token_prompt_ids:
            print(ids, id_to_token[ids]) """
    print(len(token_to_id))
    print(len(valid_tokens_for_name_func))
    fn_name, current_ids = generate_fn_name(model, encoded_token_prompt_ids, [f.name for f in functions], valid_tokens_for_name_func)
    print("Funzione scelta:", fn_name)
    current_ids = add_parameters_to_ids(model, current_ids)
    print(model.decode(current_ids))

