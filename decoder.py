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


def get_filtered_tokens_vocab(token_to_id:dict, regex: str | None = None):
    result = {}
    for token, token_id in token_to_id.items():
        if re.match(f'^[{regex}]+$', token):
            result[token_id] = token
    return result



def constrain_functions(
    model,
    encoded_token: list[int],
    valid_names: list[str],
    valid_tokens_filter_dict: dict[int, str],
    id_to_token
) -> str:
    prefix = ""
    current_ids = list(encoded_token)

    while prefix not in valid_names:
        print(prefix)
        logits = model.get_logits_from_input_ids(current_ids)

        best_token_id = None
        best_logit = float("-inf")

        for i, (tid, token) in enumerate(valid_tokens_filter_dict.items()):
            candidate = prefix + token
            """     print("candidate",candidate) if i%1000 == 0 else None """
            for name in valid_names:
                if name.startswith(candidate):
                    if logits[tid] > best_logit:
                        print("*"*50)
                        print("tid", tid, "logits[tid]",logits[tid], "value:", id_to_token[tid])
                        best_logit = logits[tid]
                        best_token_id = tid
                    break

        if best_token_id is None:
            raise ValueError(f"No valid token found at prefix: {prefix!r}")

        prefix += valid_tokens_filter_dict[best_token_id]
        current_ids.append(best_token_id)

    return prefix, current_ids

def constrain_parameters(model, encoded_tokens, valid_tokens_filter_dict, max_tokens = 20):
    
    
    best_token_id = None
    best_probability = float("-inf")
    print(valid_tokens_filter_dict)
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(encoded_tokens)

        best_token_id = max(range(len(logits)), key=lambda i: logits[i])
        print("best_choise", f"{model.decode(best_token_id)!r}")

        if best_token_id not in  valid_tokens_filter_dict:
            break

        encoded_tokens.append(best_token_id)

    return encoded_tokens






def get_regex_for_type(parameter):
    TYPE_REGEX = {
    "number": "0-9.",
    "string": "a-zA-Z",
    }

    return TYPE_REGEX[parameter.type]
    
def generate_parameters(model, encoded_tokens, parameters, token_to_id):
    tensor = model.encode(f', ')
    encoded_comma_and_space = [int(x) for x in tensor[0]]
    tensor = model.encode(f'}}}}')
    encoded_end = [int(x) for x in tensor[0]]
    tensor = model.encode(f'"')
    encoded_quotes = [int(x) for x in tensor[0]]
    for i, (key, parameter) in enumerate(parameters.items()):
        tensor = model.encode(f'"{key}": ')
        encoded_key = [int(x) for x in tensor[0]]
        encoded_tokens.extend(encoded_key)
        if parameter.type =="string":
            encoded_tokens.extend(encoded_quotes)
        valid_tokens_filter = get_regex_for_type(parameter)
        valid_tokens_filter_vocab = get_filtered_tokens_vocab(token_to_id, valid_tokens_filter)
        current_tokens = constrain_parameters(model, encoded_tokens, valid_tokens_filter_vocab)
        encoded_tokens.extend(current_tokens)
        if parameter.type =="string":
                encoded_tokens.extend(encoded_quotes)
        if i < len(parameters) -1:
            encoded_tokens.extend(encoded_comma_and_space)
    encoded_tokens.extend(encoded_end)
    return encoded_tokens

def add_parameters_to_ids(model,  current_ids: list[int]):
    output = f'", "parameters" : {{'
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
    user_prompt = "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"
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
    valid_tokens_for_name_func = get_filtered_tokens_vocab(token_to_id, "a-zA-Z0-9_")
    print(f"Token validi per nomi funzione: {len(valid_tokens_for_name_func)}")
    print(dict(list(valid_tokens_for_name_func.items())[:100]))
    print()
    """ 
        for ids in encoded_token_prompt_ids:
            print(ids, id_to_token[ids]) """
    print(len(token_to_id))
    print(len(valid_tokens_for_name_func))
    fn_name, current_ids = constrain_functions(model, encoded_token_prompt_ids, [f.name for f in functions], valid_tokens_for_name_func, id_to_token)
    print("Funzione scelta:", fn_name)
    parameters = next(f.parameters for f in functions if f.name == fn_name)
    print(parameters)
    current_ids = add_parameters_to_ids(model, current_ids)
    current_ids = generate_parameters(model, current_ids, parameters, token_to_id)
    print(model.decode(current_ids))

