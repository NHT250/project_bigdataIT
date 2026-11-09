"""Split text into overlapping token windows, keeping the original characters."""


def chunk_text(text, tokenizer, size=500, overlap=50):
    offsets = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, truncation=False
    )["offset_mapping"]
    chunks = []
    start = 0
    while start < len(offsets):
        window = offsets[start:start + size]
        chunks.append(text[window[0][0]:window[-1][1]])
        if start + size >= len(offsets):
            break
        start += size - overlap
    return chunks
