# SPDX-License-Identifier: GPL-3.0-or-later
#
# Toolify: Empower any LLM with function calling capabilities.
# Copyright (C) 2025 FunnyCups (https://github.com/funnycups)

"""
Base converter class for format conversion.
All format converters should inherit from this class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Conversion result with success status and data/error"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        """Validate conversion result"""
        if self.success and self.data is None:
            logger.warning("Successful conversion but data is None")
        if not self.success and self.error is None:
            logger.warning("Failed conversion but error is None")


class BaseConverter(ABC):
    """
    Abstract base class for format converters.
    
    Each converter handles conversion TO its target format FROM other formats.
    For example, AnthropicConverter converts FROM OpenAI/Gemini TO Anthropic.
    """
    
    def __init__(self):
        self.original_model: Optional[str] = None
        self._streaming_state = {}
    
    def set_original_model(self, model: str):
        """Set the original model name for response conversion"""
        self.original_model = model
    
    def reset_streaming_state(self):
        """Reset streaming state for new stream"""
        self._streaming_state = {}
        logger.debug(f"{self.__class__.__name__}: Streaming state reset")
    
    @abstractmethod
    def convert_request(
        self,
        data: Dict[str, Any],
        target_format: str,
        headers: Optional[Dict[str, str]] = None
    ) -> ConversionResult:
        """
        Convert request from this format to target format.
        
        Args:
            data: Request data in this converter's format
            target_format: Target format (openai, anthropic, gemini)
            headers: Optional request headers
            
        Returns:
            ConversionResult with converted data or error
        """
        pass
    
    @abstractmethod
    def convert_response(
        self,
        data: Dict[str, Any],
        source_format: str,
        target_format: str
    ) -> ConversionResult:
        """
        Convert response from source format to this converter's format.
        
        Args:
            data: Response data in source format
            source_format: Source format (openai, anthropic, gemini)
            target_format: This converter's format
            
        Returns:
            ConversionResult with converted data or error
        """
        pass
    
    def convert_streaming_chunk(
        self,
        data: Dict[str, Any],
        source_format: str
    ) -> ConversionResult:
        """
        Convert streaming response chunk.
        
        Args:
            data: Chunk data in source format
            source_format: Source format (openai, anthropic, gemini)
            
        Returns:
            ConversionResult with converted chunk or error
        """
        # Default implementation: use regular response conversion
        return self.convert_response(data, source_format, self.get_format_name())
    
    @abstractmethod
    def get_format_name(self) -> str:
        """Return the format name this converter handles"""
        pass
    
    @staticmethod
    def safe_get(data: Dict, *keys, default=None):
        """Safely get nested dictionary value"""
        current = data
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
                if current is None:
                    return default
            else:
                return default
        return current if current is not None else default

