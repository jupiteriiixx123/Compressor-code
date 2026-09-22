from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List

from .io_utils import answer_letter, normalize_ws, read_jsonl


def prepare_quality(input_path: str, split: str) -> List[Dict[str, Any]]:
    articles: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    for row in read_jsonl(input_path):
        article_id = str(row["article_id"])
        context = str(row["article"]).strip()

        if article_id not in articles:
            articles[article_id] = {
                "id": f"quality::{article_id}",
                "dataset": "quality",
                "split": split,
                "article_id": article_id,
                "context": context,
                "probes": [],
                "metadata": {
                    "title": row.get("title"),
                    "author": row.get("author"),
                    "source": row.get("source"),
                    "topic": row.get("topic"),
                    "url": row.get("url"),
                    "year": row.get("year"),
                    "license": row.get("license"),
                    "question_set_ids": [],
                },
            }
        else:
            if normalize_ws(articles[article_id]["context"]) != normalize_ws(context):
                raise ValueError(f"Article text mismatch for article_id={article_id}")

        set_id = row.get("set_unique_id")
        if set_id is not None:
            articles[article_id]["metadata"]["question_set_ids"].append(str(set_id))

        for q in row.get("questions", []):
            if not isinstance(q, dict):
                continue
            question = normalize_ws(q.get("question", ""))
            options = [normalize_ws(x) for x in q.get("options", [])]
            if not question or len(options) != 4:
                continue

            gold_label = q.get("gold_label")
            letter = answer_letter(gold_label)
            answer_index = int(gold_label) - 1 if letter is not None else None

            articles[article_id]["probes"].append({
                "question": question,
                "choices": options,
                "answer_index": answer_index,
                "answer_letter": letter,
                "difficult": q.get("difficult"),
                "source_set_unique_id": set_id,
            })

    for article in articles.values():
        seen = set()
        unique = []
        for probe in article["probes"]:
            key = (probe["question"], tuple(probe["choices"]))
            if key in seen:
                continue
            seen.add(key)
            unique.append(probe)
        for idx, probe in enumerate(unique, 1):
            probe["id"] = f"{article['id']}::q{idx:04d}"
        article["probes"] = unique

    return list(articles.values())
