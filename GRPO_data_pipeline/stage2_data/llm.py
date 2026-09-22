from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import requests


class LLMClient:
    supports_parallel: bool = False

    def generate(
        self,
        system: str,
        user: str,
        guided_json: Dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError


@dataclass
class OpenAICompatibleClient(LLMClient):
    base_url: str
    model: str
    api_key: str
    temperature: float = 0.0
    max_new_tokens: int = 4096
    timeout: int = 600
    retries: int = 4
    json_mode: bool = False
    max_token_field: str = "max_tokens"
    supports_parallel: bool = True

    def generate(
        self,
        system: str,
        user: str,
        guided_json: Dict[str, Any] | None = None,
    ) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            self.max_token_field: self.max_new_tokens,
        }

        # vLLM accepts guided_json as an extra top-level field in the raw HTTP
        # request. A supplied schema takes precedence over generic json_object
        # mode because our local format is a top-level array, not an object.
        if guided_json is not None:
            payload["guided_json"] = guided_json
        elif self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = e
                if attempt + 1 == self.retries:
                    break
                time.sleep(min(2 ** attempt, 10))

        raise RuntimeError(
            f"OpenAI-compatible request failed after {self.retries} attempts: {last_error}"
        )


class HFClient(LLMClient):
    supports_parallel = False

    def __init__(
        self,
        model_name_or_path: str,
        temperature: float = 0.0,
        max_new_tokens: int = 4096,
        max_model_tokens: int | None = None,
        attn_implementation: str = "sdpa",
        trust_remote_code: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_model_tokens = max_model_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device

    def _render(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate(
        self,
        system: str,
        user: str,
        guided_json: Dict[str, Any] | None = None,
    ) -> str:
        # Direct HF generation has no guided-decoding implementation here.
        # The prompt still requests the same JSON format.
        prompt = self._render(system, user)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        if (
            self.max_model_tokens is not None
            and prompt_tokens + self.max_new_tokens > self.max_model_tokens
        ):
            raise ValueError(
                f"Prompt would overflow model window: prompt={prompt_tokens}, "
                f"max_new={self.max_new_tokens}, max_model={self.max_model_tokens}. "
                "No silent truncation is performed."
            )

        inputs = {k: v.to(self.input_device) for k, v in inputs.items()}
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature

        with self.torch.inference_mode():
            out = self.model.generate(**inputs, **kwargs)

        generated = out[0, prompt_tokens:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["openai_compatible", "hf"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument(
        "--max-token-field",
        choices=["max_tokens", "max_completion_tokens"],
        default="max_tokens",
    )
    parser.add_argument("--max-model-tokens", type=int, default=None)
    parser.add_argument(
        "--attn-implementation",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--workers", type=int, default=1)


def build_client(args: argparse.Namespace) -> LLMClient:
    if args.backend == "openai_compatible":
        key = args.api_key or os.environ.get(args.api_key_env) or "local"
        return OpenAICompatibleClient(
            base_url=args.base_url,
            model=args.model,
            api_key=key,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            timeout=args.timeout,
            retries=args.retries,
            json_mode=args.json_mode,
            max_token_field=args.max_token_field,
        )

    if args.workers != 1:
        raise ValueError("Direct HF backend currently requires --workers 1.")
    return HFClient(
        model_name_or_path=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_model_tokens=args.max_model_tokens,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
