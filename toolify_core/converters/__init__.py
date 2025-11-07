# SPDX-License-Identifier: GPL-3.0-or-later
#
# Toolify: Empower any LLM with function calling capabilities.
# Copyright (C) 2025 FunnyCups (https://github.com/funnycups)

"""
Format converters for AI API formats.
Supports OpenAI, Anthropic, and Gemini formats.
"""

from .base_converter import BaseConverter, ConversionResult
from .converter_factory import (
    ConverterFactory,
    convert_request,
    convert_response,
    convert_streaming_chunk
)
from .openai_converter import OpenAIConverter
from .anthropic_converter import AnthropicConverter
from .gemini_converter import GeminiConverter

__all__ = [
    'BaseConverter',
    'ConversionResult',
    'ConverterFactory',
    'convert_request',
    'convert_response',
    'convert_streaming_chunk',
    'OpenAIConverter',
    'AnthropicConverter',
    'GeminiConverter',
]

