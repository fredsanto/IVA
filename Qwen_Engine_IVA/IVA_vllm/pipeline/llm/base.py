"""
pipeline/llm/base.py — Abstract LLM client base class.

All SLM access in the pipeline goes through this interface.
Tools never instantiate models directly; they call context.llm.generate().
"""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base class for all LLM clients in the pipeline."""

    @abstractmethod
    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: bool = False,  # always False in this pipeline — enforced at registry level
        repetition_penalty: float = 1.0,
        json_schema: dict | None = None,
    ) -> str:
        """
        Generate a response for a single system+user turn.

        Args:
            system:          System prompt string.
            user:            User prompt string.
            max_tokens:      Maximum new tokens to generate.
            temperature:     Sampling temperature (0.0 = greedy).
            enable_thinking: Always forced to False; parameter retained for interface
                             compatibility only.
            repetition_penalty: >1.0 penalises repeated tokens (1.0 = off); used to
                             retry a response that degenerated into a repetition loop.
            json_schema:     JSON schema the response must follow, enforced at decoding
                             time (constrained decoding); the response is then that JSON.

        Returns:
            The model's response as a plain string.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the underlying model/endpoint is ready to accept requests."""
        ...
