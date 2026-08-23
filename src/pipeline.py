import json
from typing import TYPE_CHECKING

from .constrained_decoding import constrain_functions, generate_parameters
from .models import FunctionDefinition
from .prompt import add_parameters_to_ids, build_full_prompt
from .vocab import Vocab, encode_to_ids

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


def process_prompt(
    model: "Small_LLM_Model",
    vocab: Vocab,
    functions: list[FunctionDefinition],
    formatted_functions: str,
    valid_tokens_for_name_func: dict[int, str],
    prompt_text: str,
) -> dict[str, object]:
    """Run the pipeline for one prompt, return {prompt, name, parameters}.

    Args:
        model: The LLM wrapper.
        vocab: The full tokenizer vocabulary.
        functions: The available function definitions.
        formatted_functions: Their plain-text description, primed
            into the prompt.
        valid_tokens_for_name_func: Vocab tokens eligible for
            function-name generation.
        prompt_text: The user's natural-language question.

    Returns:
        A dict with the original prompt, the chosen function name,
        and its generated parameters.
    """
    full_formatted_prompt = build_full_prompt(formatted_functions, prompt_text)
    encoded_token_prompt_ids = encode_to_ids(model, full_formatted_prompt)

    fn_name, current_ids = constrain_functions(
        model,
        encoded_token_prompt_ids,
        [f.name for f in functions],
        valid_tokens_for_name_func,
    )
    parameters = next(f.parameters for f in functions if f.name == fn_name)

    current_ids = add_parameters_to_ids(model, current_ids)
    current_ids = generate_parameters(model, current_ids, parameters, vocab)

    decoded_text = model.decode(current_ids)
    json_start = decoded_text.index('{"name"')
    parsed = json.loads(decoded_text[json_start:])

    return {
        "prompt": prompt_text,
        "name": parsed["name"],
        "parameters": parsed["parameters"],
    }
