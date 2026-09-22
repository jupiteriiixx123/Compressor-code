from __future__ import annotations

import re
from typing import Dict, List


def split_paragraphs(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts or [text.strip()]


def _split_oversized_block(block, tokenizer, chunk_tokens, overlap_tokens):
    ids = tokenizer.encode(block, add_special_tokens=False)
    if len(ids) <= chunk_tokens:
        return [block]
    step = max(1, chunk_tokens - overlap_tokens)
    pieces = []
    for start in range(0, len(ids), step):
        end = min(start + chunk_tokens, len(ids))
        pieces.append(tokenizer.decode(ids[start:end], skip_special_tokens=True).strip())
        if end >= len(ids):
            break
    return [p for p in pieces if p]


def chunk_text(text, tokenizer, chunk_tokens=1600, overlap_tokens=150) -> List[Dict]:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0")
    if not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("Require 0 <= overlap_tokens < chunk_tokens")

    paragraphs = split_paragraphs(text)
    blocks = []
    for p_idx, paragraph in enumerate(paragraphs):
        pieces = _split_oversized_block(paragraph, tokenizer, chunk_tokens, overlap_tokens)
        for piece_idx, piece in enumerate(pieces):
            blocks.append({
                "text": piece,
                "paragraph_index": p_idx,
                "piece_index": piece_idx,
                "tokens": len(tokenizer.encode(piece, add_special_tokens=False)),
            })

    chunks, current = [], []
    current_tokens = 0
    i = 0
    while i < len(blocks):
        block = blocks[i]
        sep_cost = 2 if current else 0

        if current and current_tokens + sep_cost + block["tokens"] > chunk_tokens:
            text_out = "\n\n".join(x["text"] for x in current)
            chunks.append({
                "text": text_out,
                "token_count": len(tokenizer.encode(text_out, add_special_tokens=False)),
                "paragraph_start": current[0]["paragraph_index"],
                "paragraph_end": current[-1]["paragraph_index"],
            })

            carry, carry_tokens = [], 0
            if overlap_tokens > 0:
                for old in reversed(current):
                    projected = carry_tokens + old["tokens"]
                    if carry and projected > overlap_tokens:
                        break
                    carry.insert(0, old)
                    carry_tokens = projected
                    if carry_tokens >= overlap_tokens:
                        break

            current = carry
            current_tokens = (
                len(tokenizer.encode("\n\n".join(x["text"] for x in current), add_special_tokens=False))
                if current else 0
            )

            # Important: overlap by itself may leave no room for the next block.
            # Drop the carry in that case so i can advance; otherwise this loop
            # can repeatedly emit the same chunk forever.
            next_block = blocks[i]
            sep_cost = 2 if current else 0
            if current and current_tokens + sep_cost + next_block["tokens"] > chunk_tokens:
                current = []
                current_tokens = 0
            continue

        current.append(block)
        current_tokens += sep_cost + block["tokens"]
        i += 1

    if current:
        text_out = "\n\n".join(x["text"] for x in current)
        if not chunks or text_out != chunks[-1]["text"]:
            chunks.append({
                "text": text_out,
                "token_count": len(tokenizer.encode(text_out, add_special_tokens=False)),
                "paragraph_start": current[0]["paragraph_index"],
                "paragraph_end": current[-1]["paragraph_index"],
            })

    for idx, chunk in enumerate(chunks):
        chunk["chunk_index"] = idx
        chunk["chunk_id"] = f"chunk_{idx:04d}"
    return chunks
