# ---- PATCH START ----

def decode_tokens(tokenizer, token_ids):
    tokens = [tokenizer.id_to_token(t) for t in token_ids]
    text = "".join(tokens)
    text = text.replace("Ġ", " ")
    text = text.replace("Ċ", "\n")
    return text.strip()

# ---- PATCH END ----
