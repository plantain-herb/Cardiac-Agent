def normalize_image_tokens(prompt: str, image_count: int) -> str:
    """Make the prompt image-token count match the uploaded image count."""
    if image_count == 0:
        return prompt
    if prompt.count("<image>") == image_count:
        return prompt
    clean = prompt.replace("<image>", "").strip()
    tokens = " " + ("<image>\n" * image_count)
    marker = "USER:"
    position = clean.find(marker)
    if position >= 0:
        position += len(marker)
        return clean[:position] + tokens + clean[position:].lstrip()
    return tokens.lstrip() + clean
