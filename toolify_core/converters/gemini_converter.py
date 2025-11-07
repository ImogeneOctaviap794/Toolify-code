# SPDX-License-Identifier: GPL-3.0-or-later
#
# Toolify: Empower any LLM with function calling capabilities.
# Copyright (C) 2025 FunnyCups (https://github.com/funnycups)

"""
Gemini format converter.
Handles Google Gemini API format conversions.
"""

import json
import time
import uuid
from typing import Dict, Any, Optional
import logging

from .base_converter import BaseConverter, ConversionResult

logger = logging.getLogger(__name__)


class GeminiConverter(BaseConverter):
    """Converter for Google Gemini format"""
    
    def get_format_name(self) -> str:
        return "gemini"
    
    def convert_request(
        self,
        data: Dict[str, Any],
        target_format: str,
        headers: Optional[Dict[str, str]] = None
    ) -> ConversionResult:
        """Convert Gemini request to target format"""
        try:
            if target_format == "gemini":
                return ConversionResult(success=True, data=data)
            elif target_format == "openai":
                return self._convert_to_openai_request(data)
            elif target_format == "anthropic":
                return self._convert_to_anthropic_request(data)
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
        """Convert response from source format to Gemini format"""
        try:
            if source_format == "gemini":
                return ConversionResult(success=True, data=data)
            elif source_format == "openai":
                return self._convert_from_openai_response(data)
            elif source_format == "anthropic":
                return self._convert_from_anthropic_response(data)
            else:
                return ConversionResult(
                    success=False,
                    error=f"Unsupported source format: {source_format}"
                )
        except Exception as e:
            logger.error(f"Response conversion failed: {e}")
            return ConversionResult(success=False, error=str(e))
    
    def _convert_to_openai_request(self, data: Dict) -> ConversionResult:
        """Convert Gemini request to OpenAI format"""
        openai_req = {
            "model": data.get("model", "gpt-4"),
            "messages": [],
            "stream": data.get("stream", False)
        }
        
        # Handle system instruction
        system_instruction = data.get("systemInstruction")
        if system_instruction:
            parts = system_instruction.get("parts", [])
            system_texts = [p.get("text", "") for p in parts if "text" in p]
            if system_texts:
                openai_req["messages"].append({
                    "role": "system",
                    "content": "\n".join(system_texts)
                })
        
        # Convert contents to messages
        for content in data.get("contents", []):
            # Gemini uses "user" and "model", map to OpenAI's "user" and "assistant"
            # Default to "user" if role is not specified
            gemini_role = content.get("role", "user")
            role = "user" if gemini_role == "user" else "assistant"
            parts = content.get("parts", [])
            
            # Extract text parts
            text_parts = []
            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])
                elif "functionResponse" in part:
                    # Handle function response
                    func_resp = part["functionResponse"]
                    openai_req["messages"].append({
                        "role": "tool",
                        "tool_call_id": func_resp.get("name", ""),
                        "content": json.dumps(func_resp.get("response", {}))
                    })
            
            if text_parts:
                openai_req["messages"].append({
                    "role": role,
                    "content": "\n".join(text_parts)
                })
        
        # Generation config
        gen_config = data.get("generationConfig", {})
        if "maxOutputTokens" in gen_config:
            openai_req["max_tokens"] = gen_config["maxOutputTokens"]
        if "temperature" in gen_config:
            openai_req["temperature"] = gen_config["temperature"]
        if "topP" in gen_config:
            openai_req["top_p"] = gen_config["topP"]
        if "stopSequences" in gen_config:
            openai_req["stop"] = gen_config["stopSequences"]
        
        # Convert tools
        if "tools" in data and data["tools"]:
            openai_tools = []
            for tool in data["tools"]:
                func_declarations = tool.get("functionDeclarations", [])
                for func_decl in func_declarations:
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": func_decl.get("name", ""),
                            "description": func_decl.get("description", ""),
                            "parameters": func_decl.get("parameters", {})
                        }
                    })
            if openai_tools:
                openai_req["tools"] = openai_tools
        
        return ConversionResult(success=True, data=openai_req)
    
    def _convert_to_anthropic_request(self, data: Dict) -> ConversionResult:
        """Convert Gemini request to Anthropic format"""
        anthropic_req = {
            "model": data.get("model", "claude-3-5-sonnet-20241022"),
            "messages": [],
            "max_tokens": 4096,
            "stream": data.get("stream", False)
        }
        
        # Handle system instruction
        system_instruction = data.get("systemInstruction")
        if system_instruction:
            parts = system_instruction.get("parts", [])
            system_texts = [p.get("text", "") for p in parts if "text" in p]
            if system_texts:
                anthropic_req["system"] = "\n".join(system_texts)
        
        # Convert contents to messages
        for content in data.get("contents", []):
            role = content.get("role", "user")
            parts = content.get("parts", [])
            
            # Build content array
            content_array = []
            for part in parts:
                if "text" in part:
                    content_array.append({
                        "type": "text",
                        "text": part["text"]
                    })
                elif "inline_data" in part:
                    # Handle image data
                    inline_data = part["inline_data"]
                    content_array.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": inline_data.get("mime_type", "image/png"),
                            "data": inline_data.get("data", "")
                        }
                    })
                elif "functionResponse" in part:
                    # Handle function response
                    func_resp = part["functionResponse"]
                    content_array.append({
                        "type": "tool_result",
                        "tool_use_id": func_resp.get("name", ""),
                        "content": json.dumps(func_resp.get("response", {}))
                    })
            
            if content_array:
                anthropic_req["messages"].append({
                    "role": role,
                    "content": content_array if len(content_array) > 1 else content_array[0]["text"] if content_array[0].get("type") == "text" else content_array
                })
        
        # Generation config
        gen_config = data.get("generationConfig", {})
        if "maxOutputTokens" in gen_config:
            anthropic_req["max_tokens"] = gen_config["maxOutputTokens"]
        if "temperature" in gen_config:
            anthropic_req["temperature"] = gen_config["temperature"]
        if "topP" in gen_config:
            anthropic_req["top_p"] = gen_config["topP"]
        if "stopSequences" in gen_config:
            anthropic_req["stop_sequences"] = gen_config["stopSequences"]
        
        # Convert tools
        if "tools" in data and data["tools"]:
            anthropic_tools = []
            for tool in data["tools"]:
                func_declarations = tool.get("functionDeclarations", [])
                for func_decl in func_declarations:
                    anthropic_tools.append({
                        "name": func_decl.get("name", ""),
                        "description": func_decl.get("description", ""),
                        "input_schema": func_decl.get("parameters", {})
                    })
            if anthropic_tools:
                anthropic_req["tools"] = anthropic_tools
        
        return ConversionResult(success=True, data=anthropic_req)
    
    def _convert_from_openai_response(self, data: Dict) -> ConversionResult:
        """Convert OpenAI response to Gemini format"""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")
        
        # Build parts array
        parts = []
        
        # Add text content
        if message.get("content"):
            parts.append({"text": message["content"]})
        
        # Convert tool_calls to Gemini functionCall format
        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                if tool_call.get("type") == "function":
                    function = tool_call.get("function", {})
                    try:
                        args = json.loads(function.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    
                    parts.append({
                        "functionCall": {
                            "name": function.get("name", ""),
                            "args": args
                        }
                    })
        
        # Map finish reason
        finish_reason_map = {
            "stop": "STOP",
            "length": "MAX_TOKENS",
            "tool_calls": "STOP",
            "content_filter": "SAFETY"
        }
        gemini_finish_reason = finish_reason_map.get(finish_reason, "STOP")
        
        # Token usage
        usage = data.get("usage", {})
        
        gemini_resp = {
            "candidates": [{
                "content": {
                    "parts": parts,
                    "role": "model"
                },
                "finishReason": gemini_finish_reason,
                "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0)
            },
            "modelVersion": self.original_model or data.get("model", "")
        }
        
        return ConversionResult(success=True, data=gemini_resp)
    
    def _convert_from_anthropic_response(self, data: Dict) -> ConversionResult:
        """Convert Anthropic response to Gemini format"""
        content_blocks = data.get("content", [])
        
        # Build parts array
        parts = []
        for block in content_blocks:
            if block.get("type") == "text":
                parts.append({"text": block.get("text", "")})
            elif block.get("type") == "tool_use":
                parts.append({
                    "functionCall": {
                        "name": block.get("name", ""),
                        "args": block.get("input", {})
                    }
                })
        
        # Map stop reason
        stop_reason = data.get("stop_reason", "end_turn")
        finish_reason_map = {
            "end_turn": "STOP",
            "max_tokens": "MAX_TOKENS",
            "stop_sequence": "STOP",
            "tool_use": "STOP"
        }
        gemini_finish_reason = finish_reason_map.get(stop_reason, "STOP")
        
        # Token usage
        usage = data.get("usage", {})
        
        gemini_resp = {
            "candidates": [{
                "content": {
                    "parts": parts,
                    "role": "model"
                },
                "finishReason": gemini_finish_reason,
                "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": usage.get("input_tokens", 0),
                "candidatesTokenCount": usage.get("output_tokens", 0),
                "totalTokenCount": (
                    usage.get("input_tokens", 0) + 
                    usage.get("output_tokens", 0)
                )
            },
            "modelVersion": self.original_model or data.get("model", "")
        }
        
        return ConversionResult(success=True, data=gemini_resp)

