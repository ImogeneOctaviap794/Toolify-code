# SPDX-License-Identifier: GPL-3.0-or-later
#
# Toolify: Empower any LLM with function calling capabilities.
# Copyright (C) 2025 FunnyCups (https://github.com/funnycups)

"""
OpenAI format converter.
Handles conversion from other formats to OpenAI format.
"""

import json
import time
import uuid
from typing import Dict, Any, Optional, List
import logging

from .base_converter import BaseConverter, ConversionResult

logger = logging.getLogger(__name__)


class OpenAIConverter(BaseConverter):
    """Converter for OpenAI format"""
    
    def get_format_name(self) -> str:
        return "openai"
    
    def convert_request(
        self,
        data: Dict[str, Any],
        target_format: str,
        headers: Optional[Dict[str, str]] = None
    ) -> ConversionResult:
        """
        Convert OpenAI request to target format.
        
        For OpenAI source, we convert TO other formats.
        """
        try:
            if target_format == "openai":
                # No conversion needed
                return ConversionResult(success=True, data=data)
            elif target_format == "anthropic":
                return self._convert_to_anthropic_request(data)
            elif target_format == "gemini":
                return self._convert_to_gemini_request(data)
            else:
                return ConversionResult(
                    success=False,
                    error=f"Unsupported target format: {target_format}"
                )
        except Exception as e:
            logger.error(f"Request conversion failed: {e}")
            return ConversionResult(success=False, error=str(e))
    
    def convert_response(
        self,
        data: Dict[str, Any],
        source_format: str,
        target_format: str
    ) -> ConversionResult:
        """
        Convert response from source format to OpenAI format.
        """
        try:
            if source_format == "openai":
                # No conversion needed
                return ConversionResult(success=True, data=data)
            elif source_format == "anthropic":
                return self._convert_from_anthropic_response(data)
            elif source_format == "gemini":
                return self._convert_from_gemini_response(data)
            else:
                return ConversionResult(
                    success=False,
                    error=f"Unsupported source format: {source_format}"
                )
        except Exception as e:
            logger.error(f"Response conversion failed: {e}")
            return ConversionResult(success=False, error=str(e))
    
    def _convert_to_anthropic_request(self, data: Dict) -> ConversionResult:
        """Convert OpenAI request to Anthropic format"""
        anthropic_req = {
            "model": data.get("model", "claude-3-5-sonnet-20241022"),
            "messages": [],
            "max_tokens": data.get("max_tokens", 4096),
            "stream": data.get("stream", False)
        }
        
        # Handle system message
        system_parts = []
        for msg in data.get("messages", []):
            if msg.get("role") == "system":
                system_parts.append(msg.get("content", ""))
        
        if system_parts:
            anthropic_req["system"] = "\n\n".join(system_parts)
        
        # Convert messages
        for msg in data.get("messages", []):
            if msg.get("role") == "system":
                continue  # Already handled
            
            role = msg.get("role")
            content = msg.get("content")
            
            # Handle tool results
            if role == "tool":
                # Convert tool message to Anthropic format
                tool_call_id = msg.get("tool_call_id")
                anthropic_req["messages"].append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content
                    }]
                })
            else:
                anthropic_req["messages"].append({
                    "role": role,
                    "content": content
                })
        
        # Convert tools
        if "tools" in data and data["tools"]:
            anthropic_tools = []
            for tool in data["tools"]:
                func = tool.get("function", {})
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {})
                })
            anthropic_req["tools"] = anthropic_tools
        
        # Optional parameters
        if "temperature" in data:
            anthropic_req["temperature"] = data["temperature"]
        if "top_p" in data:
            anthropic_req["top_p"] = data["top_p"]
        if "stop" in data:
            anthropic_req["stop_sequences"] = data["stop"]
        
        return ConversionResult(success=True, data=anthropic_req)
    
    def _convert_to_gemini_request(self, data: Dict) -> ConversionResult:
        """Convert OpenAI request to Gemini format"""
        gemini_req = {
            "contents": [],
        }
        
        # System instruction
        system_parts = []
        for msg in data.get("messages", []):
            if msg.get("role") == "system":
                system_parts.append(msg.get("content", ""))
        
        if system_parts:
            gemini_req["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }
        
        # Convert messages to Gemini contents
        for msg in data.get("messages", []):
            if msg.get("role") == "system":
                continue
            
            role = "user" if msg.get("role") in ["user", "tool"] else "model"
            content = msg.get("content", "")
            
            gemini_req["contents"].append({
                "role": role,
                "parts": [{"text": content}]
            })
        
        # Generation config
        generation_config = {}
        if "max_tokens" in data:
            generation_config["maxOutputTokens"] = data["max_tokens"]
        if "temperature" in data:
            generation_config["temperature"] = data["temperature"]
        if "top_p" in data:
            generation_config["topP"] = data["top_p"]
        if "stop" in data:
            generation_config["stopSequences"] = data["stop"]
        
        if generation_config:
            gemini_req["generationConfig"] = generation_config
        
        # Convert tools
        if "tools" in data and data["tools"]:
            gemini_tools = []
            for tool in data["tools"]:
                func = tool.get("function", {})
                gemini_tools.append({
                    "functionDeclarations": [{
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {})
                    }]
                })
            gemini_req["tools"] = gemini_tools
        
        return ConversionResult(success=True, data=gemini_req)
    
    def _convert_from_anthropic_response(self, data: Dict) -> ConversionResult:
        """Convert Anthropic response to OpenAI format"""
        # Build OpenAI message
        message = {"role": "assistant", "content": None}
        
        # Extract content
        content_blocks = data.get("content", [])
        text_parts = []
        tool_calls = []
        
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id", f"call_{uuid.uuid4().hex}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}))
                    }
                })
        
        if text_parts:
            message["content"] = "\n".join(text_parts)
        
        if tool_calls:
            message["tool_calls"] = tool_calls
            if not message["content"]:
                message["content"] = None
        
        # Determine finish reason
        stop_reason = data.get("stop_reason", "stop")
        finish_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls"
        }
        finish_reason = finish_reason_map.get(stop_reason, "stop")
        
        openai_resp = {
            "id": data.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.original_model or data.get("model", ""),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
                "total_tokens": (
                    data.get("usage", {}).get("input_tokens", 0) +
                    data.get("usage", {}).get("output_tokens", 0)
                )
            }
        }
        
        return ConversionResult(success=True, data=openai_resp)
    
    def _convert_from_gemini_response(self, data: Dict) -> ConversionResult:
        """Convert Gemini response to OpenAI format"""
        candidates = data.get("candidates", [])
        if not candidates:
            # No response
            return ConversionResult(
                success=False,
                error="No candidates in Gemini response"
            )
        
        candidate = candidates[0]
        content_data = candidate.get("content", {})
        parts = content_data.get("parts", [])
        
        # Extract text and function calls
        text_parts = []
        tool_calls = []
        
        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                func_call = part["functionCall"]
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": func_call.get("name", ""),
                        "arguments": json.dumps(func_call.get("args", {}))
                    }
                })
        
        # Build message
        message = {"role": "assistant"}
        if text_parts:
            message["content"] = "\n".join(text_parts)
        else:
            message["content"] = None
        
        if tool_calls:
            message["tool_calls"] = tool_calls
        
        # Determine finish reason
        finish_reason_map = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter"
        }
        gemini_reason = candidate.get("finishReason", "STOP")
        finish_reason = finish_reason_map.get(gemini_reason, "stop")
        
        # Token usage
        usage_metadata = data.get("usageMetadata", {})
        
        openai_resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.original_model or data.get("modelVersion", ""),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
                "total_tokens": usage_metadata.get("totalTokenCount", 0)
            }
        }
        
        return ConversionResult(success=True, data=openai_resp)

