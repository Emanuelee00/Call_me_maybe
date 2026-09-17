from typing import TYPE_CHECKING

from .models import ParameterSchema
from .vocab import (
    Vocab,
    encode_to_ids,
    get_filtered_tokens_vocab,
    get_stop_token_ids,
)

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


def masked_argmax(logits: list[float], candidate_ids: set[int]) -> int:
    """Pick the id with the highest logit, considering only candidate_ids.

    Args:
        logits: Raw next-token scores for the whole vocabulary.
        candidate_ids: The token ids allowed to be picked.

    Returns:
        The id in candidate_ids with the highest logit.
    """
    return max(candidate_ids, key=lambda token_id: logits[token_id])


def constrain_functions(
    model: "Small_LLM_Model",
    encoded_token: list[int],
    valid_names: list[str],
    allowed_tokens: dict[int, str],
) -> tuple[str, list[int]]:
    """Generate token by token until the result matches a valid name.

    Args:
        model: The LLM wrapper, used for its logits.
        encoded_token: Token ids of the prompt to continue from.
        valid_names: The exact strings generation must land on.
        allowed_tokens: Vocabulary tokens eligible as candidates.

    Returns:
        The matched name, and encoded_token with the generated ids
        appended.
    """
    prefix = ""
    current_ids = list(encoded_token)

    while prefix not in valid_names:
        logits = model.get_logits_from_input_ids(current_ids)

        candidate_ids = set()
        for token_id, token in allowed_tokens.items():
            candidate = prefix + token
            for name in valid_names:
                if name.startswith(candidate):
                    candidate_ids.add(token_id)
                    break

        if not candidate_ids:
            raise ValueError(f"No valid token found at prefix: {prefix!r}")

        best_token_id = masked_argmax(logits, candidate_ids)
        prefix += allowed_tokens[best_token_id]
        current_ids.append(best_token_id)

    return prefix, current_ids


def find_repeating_tail(
    tokens: list[int], max_period: int = 12, min_repeats: int = 4
) -> int | None:
    """Detect whether the tail of tokens is a block repeated min_repeats times.

    Args:
        tokens: The generated token ids so far.
        max_period: Largest block length to check for.
        min_repeats: How many consecutive copies of a block count as
            a stuck repetition loop.

    Returns:
        The block length if the tail is min_repeats consecutive
        copies of it, else None.
    """
    for period in range(1, max_period + 1):
        block_len = period * min_repeats
        if len(tokens) < block_len:
            continue
        tail = tokens[-block_len:]
        block = tail[:period]
        if all(
            tail[i * period : (i + 1) * period] == block
            for i in range(min_repeats)
        ):
            return period
    return None


def constrain_value(
    model: "Small_LLM_Model",
    encoded_tokens: list[int],
    allowed_tokens: dict[int, str],
    minus_token_id: int,
    stop_token_ids: set[int],
    max_tokens: int = 200,
) -> list[int]:
    """Generate a value masked to valid chars, until a stop token wins.

    Args:
        model: The LLM wrapper, used for its logits.
        encoded_tokens: Token ids generated so far.
        allowed_tokens: Vocabulary tokens eligible as candidates.
        minus_token_id: Id allowed only as the very first token.
        stop_token_ids: Ids that end generation when picked.
        max_tokens: Safety cap on generated tokens before giving up.

    Returns:
        The generated token ids, not including the stop token.

    Raises:
        ValueError: If max_tokens is reached without hitting a stop
            token.
    """
    generated: list[int] = []
    stopped = False

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(encoded_tokens + generated)

        candidate_ids = set(allowed_tokens)
        if not generated:
            candidate_ids.add(minus_token_id)
        if generated:
            candidate_ids |= stop_token_ids

        best_token_id = masked_argmax(logits, candidate_ids)
        if best_token_id in stop_token_ids:
            stopped = True
            break

        generated.append(best_token_id)

        period = find_repeating_tail(generated)
        if period is not None:
            # Model is stuck looping on the same block: cut it down to a
            # single copy and stop, instead of burning max_tokens on a
            # sequence that will never emit a stop token on its own.
            del generated[-(period * 3) :]
            stopped = True
            break

    if not stopped:
        raise ValueError(
            "Number too long: exceeded max_tokens without hitting stop"
        )

    return generated


def get_regex_for_type(parameter: ParameterSchema) -> str:
    """Character class for a parameter's type (digits, or any non-quote).

    Args:
        parameter: The parameter's schema, read for its type.

    Returns:
        A regex character class body matching that type's valid
        characters.
    """
    TYPE_REGEX = {
        "number": "0-9.",
        "string": '^"',
    }

    return TYPE_REGEX[parameter.type]


def generate_parameters(
    model: "Small_LLM_Model",
    encoded_tokens: list[int],
    parameters: dict[str, ParameterSchema],
    vocab: Vocab,
) -> list[int]:
    """Append each parameter's key/value to encoded_tokens, close object.

    Args:
        model: The LLM wrapper, used for encoding and logits.
        encoded_tokens: Token ids generated so far.
        parameters: The function's parameter schemas, in order.
        vocab: The full tokenizer vocabulary.

    Returns:
        encoded_tokens with every parameter's key and generated
        value appended, and the JSON object closed.
    """
    encoded_comma_and_space = encode_to_ids(model, ", ")
    encoded_end = encode_to_ids(model, "}}")
    encoded_quotes = encode_to_ids(model, '"')
    encoded_minus = encode_to_ids(model, "-")
    for i, (key, parameter) in enumerate(parameters.items()):
        encoded_key = encode_to_ids(model, f'"{key}": ')
        encoded_tokens.extend(encoded_key)

        if parameter.type == "boolean":
            boolean_vocab = get_filtered_tokens_vocab(vocab, "a-zA-Z")
            _, encoded_tokens = constrain_functions(
                model, encoded_tokens, ["true", "false"], boolean_vocab
            )
        else:
            if parameter.type == "string":
                encoded_tokens.extend(encoded_quotes)
            char_class = get_regex_for_type(parameter)
            allowed_tokens = get_filtered_tokens_vocab(vocab, char_class)
            if parameter.type == "string":
                single_stop_token_id = encoded_quotes[0]
            elif i < len(parameters) - 1:
                single_stop_token_id = encoded_comma_and_space[0]
            else:
                single_stop_token_id = encoded_end[0]
            stop_token_ids = get_stop_token_ids(vocab, single_stop_token_id)
            generated_ids = constrain_value(
                model,
                encoded_tokens,
                allowed_tokens,
                encoded_minus[0],
                stop_token_ids,
                max_tokens=200,
            )
            encoded_tokens.extend(generated_ids)
            if parameter.type == "string":
                encoded_tokens.extend(encoded_quotes)

        if i < len(parameters) - 1:
            encoded_tokens.extend(encoded_comma_and_space)
    encoded_tokens.extend(encoded_end)
    return encoded_tokens
