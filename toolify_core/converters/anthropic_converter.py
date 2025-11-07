# SPDX-License-Identifier: GPL-3.0-or-later
#
# Toolify: Empower any LLM with function calling capabilities.
# Copyright (C) 2025 FunnyCups (https://github.com/funnycups)

"""
Anthropic format converter.
Enhanced version that integrates existing anthropic_adapter functionality.
"""

import json
import time
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .base_converter import BaseConverter, ConversionResult

logger = logging.getLogger(__name__)


class AnthropicConverter(BaseConverter):
    """Converter for Anthropic format"""
    
    def get_format_name(self) -> str:
        return "anthropic"
    
    def convert_request(
        self,
        data: Dict[str, Any],
        target_format: str,
        headers: Optional[Dict[str, str]] = None
    ) -> ConversionResult:
        """Convert Anthropic request to target format"""
        try:
            if target_format == "anthropic":
                return ConversionResult(success=True, data=data)
            elif target_format == "openai":
                return self._convert_to_openai_request(data)
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
        """Convert response from source format to Anthropic format"""
        try:
            if source_format == "anthropic":
                return ConversionResult(success=True, data=data)
            elif source_format == "openai":
                return self._convert_from_openai_response(data)
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
    
    def _convert_to_openai_request(self, data: Dict) -> ConversionResult:
        """Convert Anthropic request to OpenAI format"""
        openai_req = {
            "model": data.get("model", "gpt-4"),
            "messages": [],
            "stream": data.get("stream", False)
        }
        
        # Handle system message (can be string or array with cache_control)
        if "system" in data and data["system"]:
            system_content = data["system"]
            
            if isinstance(system_content, list):
                text_parts = []
                for block in system_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                system_text = "\n\n".join(text_parts) if text_parts else ""
            else:
                system_text = system_content
            
            if system_text:
                openai_req["messages"].append({
                    "role": "system",
                    "content": system_text
                })
        
        # Convert messages
        for msg in data.get("messages", []):
            # Handle tool_result messages
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                text_parts = []
                for content_block in msg["content"]:
                    if content_block.get("type") == "text":
                        text_parts.append(content_block.get("text", ""))
                    elif content_block.get("type") == "tool_result":
                        # Convert tool_result to tool message
                        openai_req["messages"].append({
                            "role": "tool",
                            "tool_call_id": content_block.get("tool_use_id", ""),
                            "content": content_block.get("content", "")
                        })
                if text_parts:
                    openai_req["messages"].append({
                        "role": "user",
                        "content": " ".join(text_parts)
                    })
            else:
                # Regular message
                content = msg.get("content")
                if isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    content = " ".join(text_parts) if text_parts else ""
                
                openai_req["messages"].append({
                    "role": msg.get("role"),
                    "content": content
                })
        
        # Convert tools
        if "tools" in data and data["tools"]:
            openai_tools = []
            for tool in data["tools"]:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {})
                    }
                })
            openai_req["tools"] = openai_tools
        
        # Map parameters
        if "max_tokens" in data:
            openai_req["max_tokens"] = data["max_tokens"]
        if "temperature" in data:
            openai_req["temperature"] = data["temperature"]
        if "top_p" in data:
            openai_req["top_p"] = data["top_p"]
        if "stop_sequences" in data:
            openai_req["stop"] = data["stop_sequences"]
        
        return ConversionResult(success=True, data=openai_req)
    
    def _convert_to_gemini_request(self, data: Dict) -> ConversionResult:
        """Convert Anthropic request to Gemini format"""
        gemini_req = {"contents": []}
        
        # System instruction
        if "system" in data and data["system"]:
            system_content = data["system"]
            if isinstance(system_content, list):
                text_parts = []
                for block in system_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                system_text = "\n\n".join(text_parts)
            else:
                system_text = system_content
            
            if system_text:
                gemini_req["systemInstruction"] = {
                    "parts": [{"text": system_text}]
                }
        
        # Convert messages
        for msg in data.get("messages", []):
            role = "user" if msg.get("role") == "user" else "model"
            content = msg.get("content")
            
            if isinstance(content, list):
                # Handle complex content
                parts = []
                for block in content:
                    if block.get("type") == "text":
                        parts.append({"text": block.get("text", "")})
                    elif block.get("type") == "image":
                        # Handle image content
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            parts.append({
                                "inline_data": {
                                    "mime_type": source.get("media_type", "image/png"),
                                    "data": source.get("data", "")
                                }
                            })
                
                if parts:
                    gemini_req["contents"].append({"role": role, "parts": parts})
            else:
                # Simple text content
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
        if "stop_sequences" in data:
            generation_config["stopSequences"] = data["stop_sequences"]
        
        if generation_config:
            gemini_req["generationConfig"] = generation_config
        
        # Convert tools
        if "tools" in data and data["tools"]:
            gemini_tools = []
            for tool in data["tools"]:
                gemini_tools.append({
                    "functionDeclarations": [{
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {})
                    }]
                })
            gemini_req["tools"] = gemini_tools
        
        return ConversionResult(success=True, data=gemini_req)
    
    def _convert_from_openai_response(self, data: Dict) -> ConversionResult:
        """Convert OpenAI response to Anthropic format"""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")
        
        # Build content array
        content = []
        
        # Add text content
        if message.get("content"):
            content.append({
                "type": "text",
                "text": message.get("content")
            })
        
        # Convert tool_calls to Anthropic tool_use format
        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                if tool_call.get("type") == "function":
                    function = tool_call.get("function", {})
                    try:
                        args = json.loads(function.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    
                    content.append({
                        "type": "tool_use",
                        "id": tool_call.get("id", f"toolu_{uuid.uuid4().hex}"),
                        "name": function.get("name", ""),
                        "input": args
                    })
        
        # If no content, add empty text
        if not content:
            content.append({"type": "text", "text": ""})
        
        # Map finish_reason
        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "end_turn"
        }
        stop_reason = stop_reason_map.get(finish_reason, "end_turn")
        
        anthropic_resp = {
            "id": data.get("id", f"msg_{uuid.uuid4().hex}"),
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": self.original_model or data.get("model", ""),
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": data.get("usage", {}).get("completion_tokens", 0)
            }
        }
        
        return ConversionResult(success=True, data=anthropic_resp)
    
    def _convert_from_gemini_response(self, data: Dict) -> ConversionResult:
        """Convert Gemini response to Anthropic format"""
        candidates = data.get("candidates", [])
        if not candidates:
            return ConversionResult(
                success=False,
                error="No candidates in Gemini response"
            )
        
        candidate = candidates[0]
        content_data = candidate.get("content", {})
        parts = content_data.get("parts", [])
        
        # Build content array
        content = []
        for part in parts:
            if "text" in part:
                content.append({
                    "type": "text",
                    "text": part["text"]
                })
            elif "functionCall" in part:
                func_call = part["functionCall"]
                content.append({
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex}",
                    "name": func_call.get("name", ""),
                    "input": func_call.get("args", {})
                })
        
        if not content:
            content.append({"type": "text", "text": ""})
        
        # Map finish reason
        finish_reason_map = {
            "STOP": "end_turn",
            "MAX_TOKENS": "max_tokens",
            "SAFETY": "end_turn",
            "RECITATION": "end_turn"
        }
        gemini_reason = candidate.get("finishReason", "STOP")
        stop_reason = finish_reason_map.get(gemini_reason, "end_turn")
        
        # Token usage
        usage_metadata = data.get("usageMetadata", {})
        
        anthropic_resp = {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": self.original_model or data.get("modelVersion", ""),
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": usage_metadata.get("promptTokenCount", 0),
                "output_tokens": usage_metadata.get("candidatesTokenCount", 0)
            }
        }
        
        return ConversionResult(success=True, data=anthropic_resp)

