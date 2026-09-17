import argparse
import json
import os
import sys

from llm_sdk import Small_LLM_Model

from .models import FunctionDefinition, Prompt
from .pipeline import process_prompt
from .prompt import build_system_prompt, format_functions_list
from .utils import load_and_validate_definitions
from .vocab import get_filtered_tokens_vocab, load_vocab


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments: functions_definition, input, output paths.

    --input-prompt, when given, overrides --input with a single
    prompt passed directly on the command line.

    Returns:
        The parsed arguments, with data/ defaults applied.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input", default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output", default="data/output/function_calling_results.json"
    )
    parser.add_argument(
        "--input-prompt",
        default=None,
        help="Process this single prompt instead of reading --input.",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    """Run every test prompt through the pipeline, write results as JSON."""
    args = parse_arguments()

    try:
        functions = load_and_validate_definitions(
            args.functions_definition, FunctionDefinition
        )
        if args.input_prompt is not None:
            prompts = [Prompt(prompt=args.input_prompt)]
        else:
            prompts = load_and_validate_definitions(args.input, Prompt)
    except ValueError as error:
        print(f"Error: {error}")
        sys.exit(1)

    functions_list = build_system_prompt(functions)
    formatted_functions = format_functions_list(functions_list)

    model = Small_LLM_Model()
    path = model.get_path_to_vocab_file()
    vocab = load_vocab(path)
    valid_tokens_for_name_func = get_filtered_tokens_vocab(vocab, "a-zA-Z0-9_")

    results = []
    for prompt in prompts:
        try:
            result = process_prompt(
                model,
                vocab,
                functions,
                formatted_functions,
                valid_tokens_for_name_func,
                prompt.prompt,
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
