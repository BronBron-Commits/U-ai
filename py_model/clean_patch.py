def clean_text(text):
    text = text.replace("Ġ", " ")
    text = text.replace("Ċ", "\n")
    return text.strip()
