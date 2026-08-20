import argparse
import json
from pydantic import BaseModel
from models import FunctionDefinition, Prompt
from typing import Type
from pprint import pprint
from utils import load_and_validate_definitions


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition", default="data/input/functions_definition.json"
    )
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    functions = load_and_validate_definitions(
        args.functions_definition, FunctionDefinition
    )
    """ for item in functions:
        print(item)
       
    prompts = load_and_validate_definitions(args.input, Prompt)
    for prompt in prompts:
        print (prompt) """

    # print((functions), len(prompts))

    print(build_system_prompt(functions))


if __name__ == "__main__":
    main()
