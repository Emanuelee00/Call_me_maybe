import json
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


class Vocab(BaseModel):
    """The tokenizer's vocabulary, both directions.

    Attributes:
        token_to_id: Vocab token text mapped to its id.
        id_to_token: The reverse mapping, id to token text.
    """

    token_to_id: dict[str, int]
    id_to_token: dict[int, str]


def load_vocab(path: str) -> Vocab:
    """Load the tokenizer vocab file and return it as a Vocab.

    Args:
        path: Path to the vocab JSON file (token text -> id).

    Returns:
        The loaded Vocab, with both directions of the mapping.
    """
    with open(path) as f:
        token_to_id = json.load(f)
    id_to_token = {v: k for k, v in token_to_id.items()}

    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token)


def get_filtered_tokens_vocab(
    vocab: Vocab, regex: str | None = None
) -> dict[int, str]:
    """Keep only tokens whose whole text matches the character class.

    Args:
        vocab: The full tokenizer vocabulary.
        regex: A character class body, e.g. "0-9." or "a-zA-Z".

    Returns:
        The subset of id -> token pairs whose token text is made
        entirely of characters from that class.
    """
    result = {}
    for token, token_id in vocab.token_to_id.items():
        if re.match(f"^[{regex}]+$", token):
            result[token_id] = token
    return result


def encode_to_ids(model: "Small_LLM_Model", text: str) -> list[int]:
    """Encode text and return it as a plain list of int token ids.

    Args:
        model: The LLM wrapper, used for its tokenizer.
        text: The text to tokenize.

    Returns:
        The token ids for text, as a plain list of int.
    """
    tensor = model.encode(text)
    ids = [int(x) for x in tensor[0]]
    return ids


def get_stop_token_ids(vocab: Vocab, stop_token_id: int) -> set[int]:
    """Every token that starts with the same text as stop_token_id.

    Args:
        vocab: The full tokenizer vocabulary.
        stop_token_id: Id of the reference token to match against.

    Returns:
        Every token id whose text starts with stop_token_id's text
        (including stop_token_id itself).
    """
    prefix = vocab.id_to_token[stop_token_id]
    return {
        token_id
        for token, token_id in vocab.token_to_id.items()
        if token.startswith(prefix)
    }
