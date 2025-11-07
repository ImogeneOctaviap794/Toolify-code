# SPDX-License-Identifier: GPL-3.0-or-later
#
# Toolify: Empower any LLM with function calling capabilities.
# Copyright (C) 2025 FunnyCups (https://github.com/funnycups)

"""
Converter factory for creating and managing format converters.
"""

from typing import Dict, Optional
import logging
from .base_converter import BaseConverter, ConversionResult

logger = logging.getLogger(__name__)


class ConverterFactory:
    """Factory for creating and managing format converters"""
    
    _converters: Dict[str, BaseConverter] = {}
    _converter_classes: Dict[str, type] = {}
    
    @classmethod
    def register_converter(cls, format_name: str, converter_class: type):
        """
        Register a converter class for a format.
        
        Args:
            format_name: Format identifier (openai, anthropic, gemini)
            converter_class: Converter class to register
        """
        cls._converter_classes[format_name] = converter_class
        logger.info(f"Registered converter for format: {format_name}")
    
    @classmethod
    def get_converter(cls, format_name: str) -> Optional[BaseConverter]:
        """
        Get or create converter for specified format.
        
        Args:
            format_name: Format identifier
            
        Returns:
            Converter instance or None if not found
        """
        if format_name not in cls._converters:
            converter_class = cls._converter_classes.get(format_name)
            if converter_class:
                cls._converters[format_name] = converter_class()
                logger.debug(f"Created new converter instance for: {format_name}")
            else:
                logger.error(f"No converter registered for format: {format_name}")
                return None
        
        return cls._converters[format_name]
    
    @classmethod
    def get_supported_formats(cls) -> list:
        """Get list of supported formats"""
        return list(cls._converter_classes.keys())
    
    @classmethod
    def is_format_supported(cls, format_name: str) -> bool:
        """Check if format is supported"""
        return format_name in cls._converter_classes
    
    @classmethod
    def clear_converters(cls):
        """Clear all converter instances (useful for testing)"""
        cls._converters.clear()
        logger.debug("Cleared all converter instances")


# Convenience functions for direct conversion

def convert_request(
    source_format: str,
    target_format: str,
    data: Dict,
    headers: Optional[Dict] = None
) -> ConversionResult:
    """
    Convert request from source format to target format.
    
    Args:
        source_format: Source API format
        target_format: Target API format
        data: Request data
        headers: Optional headers
        
    Returns:
        ConversionResult
    """
    converter = ConverterFactory.get_converter(source_format)
    if not converter:
        return ConversionResult(
            success=False,
            error=f"Unsupported source format: {source_format}"
        )
    
    # Set original model if present
    if hasattr(converter, 'set_original_model') and 'model' in data:
        converter.set_original_model(data['model'])
    
    return converter.convert_request(data, target_format, headers)


def convert_response(
    source_format: str,
    target_format: str,
    data: Dict,
    original_model: Optional[str] = None
) -> ConversionResult:
    """
    Convert response from source format to target format.
    
    Args:
        source_format: Source API format
        target_format: Target API format
        data: Response data
        original_model: Original model name from request
        
    Returns:
        ConversionResult
    """
    converter = ConverterFactory.get_converter(target_format)
    if not converter:
        return ConversionResult(
            success=False,
            error=f"Unsupported target format: {target_format}"
        )
    
    # Set original model if provided
    if hasattr(converter, 'set_original_model') and original_model:
        converter.set_original_model(original_model)
    
    return converter.convert_response(data, source_format, target_format)


def convert_streaming_chunk(
    source_format: str,
    target_format: str,
    data: Dict,
    original_model: Optional[str] = None
) -> ConversionResult:
    """
    Convert streaming chunk from source format to target format.
    
    Args:
        source_format: Source API format
        target_format: Target API format  
        data: Chunk data
        original_model: Original model name from request
        
    Returns:
        ConversionResult
    """
    logger.debug(f"Converting streaming chunk: {source_format} -> {target_format}")
    
    # Same format passthrough
    if source_format == target_format:
        logger.debug(f"Same format, returning data as-is")
        return ConversionResult(success=True, data=data)
    
    converter = ConverterFactory.get_converter(target_format)
    if not converter:
        return ConversionResult(
            success=False,
            error=f"Unsupported target format: {target_format}"
        )
    
    # Set original model if provided
    if hasattr(converter, 'set_original_model') and original_model:
        converter.set_original_model(original_model)
    
    # Check for stream-specific method
    if hasattr(converter, 'convert_streaming_chunk'):
        return converter.convert_streaming_chunk(data, source_format)
    else:
        # Fallback to regular response conversion
        return converter.convert_response(data, source_format, target_format)

