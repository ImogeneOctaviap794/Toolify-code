# SPDX-License-Identifier: GPL-3.0-or-later
#
# Toolify: Empower any LLM with function calling capabilities.
# Copyright (C) 2025 FunnyCups (https://github.com/funnycups)

"""
Reasoning budget converter.
Converts between OpenAI reasoning_effort levels and Anthropic/Gemini thinking token budgets.
"""

from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ReasoningBudgetConverter:
    """
    Converter for reasoning budget between different providers.
    
    OpenAI uses reasoning_effort: "low", "medium", "high"
    Anthropic/Gemini use thinkingBudget: number of tokens
    """
    
    # Default mappings (can be overridden in config)
    DEFAULT_OPENAI_TO_ANTHROPIC = {
        "low": 2048,
        "medium": 8192,
        "high": 16384
    }
    
    DEFAULT_OPENAI_TO_GEMINI = {
        "low": 2048,
        "medium": 8192,
        "high": 16384
    }
    
    DEFAULT_ANTHROPIC_TO_OPENAI_THRESHOLDS = {
        "low": 2048,      # tokens <= 2048 = low
        "high": 16384     # tokens >= 16384 = high, otherwise medium
    }
    
    DEFAULT_GEMINI_TO_OPENAI_THRESHOLDS = {
        "low": 2048,
        "high": 16384
    }
    
    def __init__(
        self,
        openai_to_anthropic_map: Optional[Dict[str, int]] = None,
        openai_to_gemini_map: Optional[Dict[str, int]] = None,
        anthropic_to_openai_thresholds: Optional[Dict[str, int]] = None,
        gemini_to_openai_thresholds: Optional[Dict[str, int]] = None
    ):
        """
        Initialize converter with custom mappings.
        
        Args:
            openai_to_anthropic_map: Mapping from OpenAI effort levels to Anthropic tokens
            openai_to_gemini_map: Mapping from OpenAI effort levels to Gemini tokens
            anthropic_to_openai_thresholds: Thresholds for converting Anthropic tokens to OpenAI levels
            gemini_to_openai_thresholds: Thresholds for converting Gemini tokens to OpenAI levels
        """
        self._openai_to_anthropic_map = openai_to_anthropic_map or self.DEFAULT_OPENAI_TO_ANTHROPIC
        self._openai_to_gemini_map = openai_to_gemini_map or self.DEFAULT_OPENAI_TO_GEMINI
        self._anthropic_to_openai_thresholds = anthropic_to_openai_thresholds or self.DEFAULT_ANTHROPIC_TO_OPENAI_THRESHOLDS
        self._gemini_to_openai_thresholds = gemini_to_openai_thresholds or self.DEFAULT_GEMINI_TO_OPENAI_THRESHOLDS
    
    def openai_to_anthropic(self, reasoning_effort: str) -> int:
        """
        Convert OpenAI reasoning_effort to Anthropic thinking token budget.
        
        Args:
            reasoning_effort: OpenAI effort level ("low", "medium", "high")
            
        Returns:
            Number of thinking tokens for Anthropic
        """
        tokens = self._openai_to_anthropic_map.get(reasoning_effort.lower())
        if tokens is None:
            logger.warning(f"Unknown reasoning_effort: {reasoning_effort}, using medium")
            tokens = self._openai_to_anthropic_map.get("medium", 8192)
        
        logger.debug(f"Converted OpenAI reasoning_effort '{reasoning_effort}' to Anthropic {tokens} tokens")
        return tokens
    
    def openai_to_gemini(self, reasoning_effort: str) -> int:
        """
        Convert OpenAI reasoning_effort to Gemini thinking token budget.
        
        Args:
            reasoning_effort: OpenAI effort level ("low", "medium", "high")
            
        Returns:
            Number of thinking tokens for Gemini
        """
        tokens = self._openai_to_gemini_map.get(reasoning_effort.lower())
        if tokens is None:
            logger.warning(f"Unknown reasoning_effort: {reasoning_effort}, using medium")
            tokens = self._openai_to_gemini_map.get("medium", 8192)
        
        logger.debug(f"Converted OpenAI reasoning_effort '{reasoning_effort}' to Gemini {tokens} tokens")
        return tokens
    
    def anthropic_to_openai(self, thinking_budget: int) -> str:
        """
        Convert Anthropic thinking token budget to OpenAI reasoning_effort.
        
        Args:
            thinking_budget: Number of thinking tokens
            
        Returns:
            OpenAI effort level ("low", "medium", "high")
        """
        if thinking_budget <= self._anthropic_to_openai_thresholds["low"]:
            effort = "low"
        elif thinking_budget >= self._anthropic_to_openai_thresholds["high"]:
            effort = "high"
        else:
            effort = "medium"
        
        logger.debug(f"Converted Anthropic {thinking_budget} tokens to OpenAI reasoning_effort '{effort}'")
        return effort
    
    def gemini_to_openai(self, thinking_budget: int) -> str:
        """
        Convert Gemini thinking token budget to OpenAI reasoning_effort.
        
        Args:
            thinking_budget: Number of thinking tokens
            
        Returns:
            OpenAI effort level ("low", "medium", "high")
        """
        if thinking_budget <= self._gemini_to_openai_thresholds["low"]:
            effort = "low"
        elif thinking_budget >= self._gemini_to_openai_thresholds["high"]:
            effort = "high"
        else:
            effort = "medium"
        
        logger.debug(f"Converted Gemini {thinking_budget} tokens to OpenAI reasoning_effort '{effort}'")
        return effort
    
    def convert_reasoning_param(
        self,
        source_format: str,
        target_format: str,
        value: Union[str, int]
    ) -> Optional[Union[str, int]]:
        """
        Universal converter that handles any direction.
        
        Args:
            source_format: Source API format ("openai", "anthropic", "gemini")
            target_format: Target API format
            value: Reasoning parameter value (effort level or token count)
            
        Returns:
            Converted value or None if no conversion needed
        """
        if source_format == target_format:
            return value
        
        # OpenAI -> Others
        if source_format == "openai":
            if target_format == "anthropic":
                return self.openai_to_anthropic(value)
            elif target_format == "gemini":
                return self.openai_to_gemini(value)
        
        # Others -> OpenAI
        elif target_format == "openai":
            if source_format == "anthropic":
                return self.anthropic_to_openai(value)
            elif source_format == "gemini":
                return self.gemini_to_openai(value)
        
        # Anthropic <-> Gemini (direct token mapping)
        elif source_format in ["anthropic", "gemini"] and target_format in ["anthropic", "gemini"]:
            # Direct token-to-token mapping
            return value
        
        logger.warning(f"Unsupported conversion: {source_format} -> {target_format}")
        return None


# Global converter instance
_global_converter: Optional[ReasoningBudgetConverter] = None


def get_global_converter() -> ReasoningBudgetConverter:
    """Get the global reasoning budget converter"""
    global _global_converter
    if _global_converter is None:
        _global_converter = ReasoningBudgetConverter()
    return _global_converter


def set_global_converter(converter: ReasoningBudgetConverter):
    """Set the global reasoning budget converter"""
    global _global_converter
    _global_converter = converter

