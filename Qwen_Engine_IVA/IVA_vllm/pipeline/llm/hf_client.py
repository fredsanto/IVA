"""
pipeline/llm/hf_client.py — HuggingFace pipeline LLM client (fallback / dev mode).

Sources:
  - LLM class from retrieval_agent.py (model loading, generate logic)
  - model-loading block from server_main.py (quantization, device_map, bfloat16)

Use when:
  - No GPU / no vLLM server available
  - Local development and tool testing

Set LLM_BACKEND=hf in config to activate. Slower than VLLMClient but
identical pipeline behaviour.
"""

from __future__ import annotations

from pipeline.llm.base import LLMClient


class HFClient(LLMClient):
    """
    HuggingFace transformers.pipeline("text-generation") LLM client.

    Lazy-imports torch and transformers so that the module can be imported on
    systems where those packages are not installed (e.g. CI, import-only checks).
    The actual model is loaded on first instantiation.

    enable_thinking is always silently forced to False per pipeline design rules.
    """

    def __init__(
        self,
        model_id: str  = "Qwen/Qwen3.5-9B",
        use_4bit: bool = False,
        **kwargs,
    ):
        # Deferred imports — torch/transformers only required at runtime
        import os
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            pipeline as hf_pipeline,
        )

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self.model_id   = model_id
        device          = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype     = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Optional 4-bit quantization (from server_main.py)
        quantization_config = None
        if use_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
        )
        model.eval()

        self._pipe = hf_pipeline("text-generation", model=model, tokenizer=self.tokenizer)

    # ── Private generation helpers (from server_main.py) ─────────────────────

    def _generate_from_messages(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        """
        Internal helper — apply chat template + run HF pipeline.
        enable_thinking is always forced to False (never passed through).
        Logic preserved verbatim from retrieval_agent.LLM.generate() and
        server_main.generate_text_from_messages().
        """
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,   # always False — enforced here
        )
        out = self._pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]
        return out[len(prompt):].strip()

    # ── LLMClient interface ───────────────────────────────────────────────────

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: bool = False,  # silently ignored — always False
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        do_sample = temperature > 0.0
        return self._generate_from_messages(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

    def is_available(self) -> bool:
        return self._pipe is not None
