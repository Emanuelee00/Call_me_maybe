from llm_sdk import Small_LLM_Model
import json


def get_vocabulary(path: str) -> dict[int, str]:

    with open(path, "r") as p:
        vocab = json.load(p)

    for i, (key_token, value_token) in enumerate(vocab.items()):
        if i >= 50:
            break
        print(repr(key_token), value_token)

    return vocab


def encode(input: str, vocab: dict[int, str]):
    tokens = []
    i = 0

    new_str = ""
    for c in input:
        if c == " ":
            new_str += "Ġ"
        else:
            new_str += c
    input = new_str

    while i < len(input):
        for lenght in range(len(input), 0, -1):
            valid = input[i:lenght]
            if valid in vocab:
                tokens.append(vocab[valid])
                break
        i = lenght

    return tokens


model = Small_LLM_Model()

prompt = "What is the sum of 2 and 3?"

# _encode/_decode sono privati: comodi per un test rapido, ma VIETATI nel
# progetto finale (lì bisogna usare il proprio tokenize/detokenize, come in es1).
# ids = model._encode(prompt)[0].tolist()
# print(ids)
path = model.get_path_to_vocab_file()
print(path)


vocab = get_vocabulary(path)


encoded_prompt = encode(prompt, vocab)


def decode(encoded_input: list[int], vocab: dict[int, str]):
    id_to_token = {v: k for k, v in vocab.items()}
    decoded_tokens: str = ""
    for token in encoded_input:
        decoded_tokens += id_to_token[token]
    decoded_tokens = decoded_tokens.replace("Ġ", " ")
    return decoded_tokens


decoded_prompt = decode(encoded_prompt, vocab)

print("encoded prompt", encoded_prompt)


for i in range(20):
    print("-" * 10, i, "-" * 10)
    logit = model.get_logits_from_input_ids(encoded_prompt)
    next_id = logit.index(max(logit))
    print(next_id)
    print(decode([next_id], vocab))

# print(ids)
print(decode(ids, vocab))
print(decoded_prompt)
# encoded_result = model.get_logits_from_input_ids(encoded_prompt)
# print(result)

# for _ in range(30):
#     logits = model.get_logits_from_input_ids(ids)
#     next_id = logits.index(max(logits))
#     ids.append(next_id)

# print(model._decode(ids))
