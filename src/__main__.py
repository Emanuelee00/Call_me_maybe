import argparse
import json
import os
import sys

from llm_sdk import Small_LLM_Model

from .decoder import (
    build_system_prompt,
    format_output,
    get_filtered_tokens_vocab,
    load_vocab,
    process_prompt,
)
from .models import FunctionDefinition, Prompt
from .utils import load_and_validate_definitions


def parse_arguments() -> argparse.Namespace:
    """Parse --functions_definition, --input and --output, with data/ defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition", default="data/input/functions_definition.json"
    )
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")
    args = parser.parse_args()
    return args


def main() -> None:
    """Run every test prompt through the pipeline and write the results as JSON."""
    args = parse_arguments()

    try:
        functions = load_and_validate_definitions(
            args.functions_definition, FunctionDefinition
        )
        prompts = load_and_validate_definitions(args.input, Prompt)
    except ValueError as error:
        print(f"Error: {error}")
        sys.exit(1)

    functions_list = build_system_prompt(functions)
    formatted_functions = format_output(functions_list)

    model = Small_LLM_Model()
    path = model.get_path_to_vocab_file()
    token_to_id, id_to_token = load_vocab(path)
    valid_tokens_for_name_func = get_filtered_tokens_vocab(token_to_id, "a-zA-Z0-9_")

    results = []
    for prompt in prompts:
        try:
            result = process_prompt(
                model, token_to_id, functions, formatted_functions, valid_tokens_for_name_func, prompt.prompt
            )
        except Exception as error:
            print(f"Skipping prompt {prompt.prompt!r}: {error}")
            continue
        print(result)
        results.append(result)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
