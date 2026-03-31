"""
Gemini LLM Client - Answer Generation with Google's Gemini

This module handles interaction with the Gemini LLM for answer generation.

=== WHY GEMINI IS SUITABLE FOR ENTERPRISE RAG ===

1. STRONG INSTRUCTION FOLLOWING
   - Gemini follows system prompts reliably
   - Critical for "only use context" instructions
   - Better at refusing to answer than some alternatives

2. LARGE CONTEXT WINDOW
   - Gemini 1.5 Pro: Up to 1M tokens context
   - Can include many retrieved chunks
   - Room for conversation history

3. SAFETY FEATURES
   - Built-in content filtering
   - Enterprise-appropriate responses
   - Configurable safety settings

4. COST-EFFECTIVE
   - Competitive pricing
   - Good balance of quality and cost
   - Free tier for development

=== DIFFERENCES FROM OPENAI-STYLE APIs ===

1. MESSAGE FORMAT
   - OpenAI: {"role": "system", "content": "..."}
   - Gemini: Uses Content objects with Parts

2. SYSTEM PROMPT
   - OpenAI: Separate system message
   - Gemini: system_instruction parameter

3. SAFETY SETTINGS
   - OpenAI: Limited control
   - Gemini: Granular safety category controls

4. STREAMING
   - Both support streaming
   - Gemini uses generate_content_async with stream=True
"""

import time
from typing import List, Optional, Dict, Any, Generator
from dataclasses import dataclass
import asyncio

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from app.core.reasoning.prompt_templates import PromptTemplates, FormattedPrompt
from app.config import get_settings
from app.utils.exceptions import LLMError
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """
    Complete response from the LLM with metadata.
    
    Includes performance metrics for monitoring.
    """
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model: str
    finish_reason: str


class GeminiLLMClient:
    """
    Client for generating answers using Google's Gemini LLM.
    
    Features:
    - Strict RAG prompting
    - Token counting and monitoring
    - Latency tracking
    - Configurable safety settings
    - Error handling and retries
    
    Usage:
        client = GeminiLLMClient()
        response = client.generate_answer(
            question="What is the refund policy?",
            context_chunks=["Policy states...", "Refunds are..."]
        )
        print(response.answer)
    """
    
    # Safety settings for enterprise use
    # Block only high-probability harmful content
    SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize Gemini LLM client.
        
        Args:
            api_key: Google API key. If None, loaded from config.
            model: Model name. If None, loaded from config.
        """
        settings = get_settings()
        
        self.api_key = api_key or settings.google_api_key
        self.model_name = model or settings.gemini_model
        self.max_output_tokens = settings.max_output_tokens
        self.temperature = settings.temperature
        
        if not self.api_key:
            raise LLMError(
                "Google API key not configured",
                details={"hint": "Set GOOGLE_API_KEY in .env file"}
            )
        
        # Configure API
        genai.configure(api_key=self.api_key)
        
        # Initialize model with system instruction
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=PromptTemplates.get_system_prompt(),
            safety_settings=self.SAFETY_SETTINGS,
            generation_config=genai.GenerationConfig(
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature
            )
        )
        
        logger.info(
            "GeminiLLMClient initialized",
            model=self.model_name,
            temperature=self.temperature
        )
    
    def generate_answer(
        self,
        question: str,
        context_chunks: List[str],
        conversation_history: Optional[List[dict]] = None
    ) -> LLMResponse:
        """
        Generate an answer using RAG with Gemini.
        
        Args:
            question: User's question
            context_chunks: Retrieved document chunks
            conversation_history: Optional previous turns
            
        Returns:
            LLMResponse with answer and metrics
            
        Raises:
            LLMError: If generation fails
        """
        start_time = time.time()
        
        # Format the prompt
        formatted = PromptTemplates.format_rag_prompt(
            question=question,
            context_chunks=context_chunks,
            conversation_history=conversation_history
        )
        
        logger.info(
            "Generating answer",
            question_length=len(question),
            context_length=formatted.context_length,
            chunks=formatted.chunk_count
        )
        
        try:
            # Generate response
            response = self.model.generate_content(formatted.user_prompt)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract token counts
            # Note: Gemini provides usage metadata
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count
            
            # Get finish reason
            finish_reason = "STOP"
            if response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)
            
            # Extract text
            answer = response.text
            
            logger.info(
                "Answer generated",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=round(latency_ms, 2)
            )
            
            return LLMResponse(
                answer=answer,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                model=self.model_name,
                finish_reason=finish_reason
            )
            
        except Exception as e:
            logger.error(
                "LLM generation failed",
                error=str(e),
                question=question[:100]
            )
            raise LLMError(
                f"Failed to generate answer: {str(e)}",
                details={"question": question[:100], "error": str(e)}
            )
    
    async def agenerate_answer(
        self,
        question: str,
        context_chunks: List[str],
        conversation_history: Optional[List[dict]] = None
    ) -> LLMResponse:
        """Async version of generate_answer."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_answer(question, context_chunks, conversation_history)
        )
    
    def generate_streaming(
        self,
        question: str,
        context_chunks: List[str]
    ) -> Generator[str, None, None]:
        """
        Generate answer with streaming response.
        
        Yields text chunks as they're generated.
        Useful for real-time UI updates.
        
        Args:
            question: User's question
            context_chunks: Retrieved document chunks
            
        Yields:
            Text chunks of the response
        """
        formatted = PromptTemplates.format_rag_prompt(
            question=question,
            context_chunks=context_chunks
        )
        
        try:
            response = self.model.generate_content(
                formatted.user_prompt,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error("Streaming generation failed", error=str(e))
            raise LLMError(
                f"Streaming generation failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Useful for:
        - Estimating costs
        - Checking context limits
        - Monitoring usage
        """
        try:
            result = self.model.count_tokens(text)
            return result.total_tokens
        except Exception:
            # Fallback: rough estimate (4 chars per token)
            return len(text) // 4
