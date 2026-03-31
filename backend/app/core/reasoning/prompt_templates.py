"""
Prompt Engineering - Strict RAG Prompts to Prevent Hallucination

This module contains carefully crafted prompts that force the LLM
to answer ONLY from provided context.

=== WHY PROMPT CONTROL IS NECESSARY EVEN WITH RAG ===

PROBLEM: RAG provides context, but LLMs can still:
1. Ignore the context and use training knowledge
2. "Fill in gaps" with plausible but wrong information
3. Synthesize answers that go beyond what documents say
4. Confuse information between different sources

SOLUTION: Strict prompting that:
1. Explicitly states to use ONLY provided documents
2. Requires admitting "I don't know" when info is missing
3. Requires citing sources for every claim
4. Penalizes making up information

=== PROMPT ENGINEERING PRINCIPLES ===

1. BE EXPLICIT
   - Don't assume the LLM knows what you want
   - State rules clearly and repeatedly
   - Give examples of good and bad responses

2. SET BOUNDARIES
   - Define what the LLM should NOT do
   - Explain consequences of breaking rules
   - Make the safe choice (saying "I don't know") easy

3. STRUCTURE OUTPUT
   - Request specific format
   - Ask for citations
   - Separate answer from reasoning

4. TEST ADVERSARIALLY
   - Try questions not in documents
   - Try questions that span multiple docs
   - Try trick questions that seem answerable
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class FormattedPrompt:
    """Container for formatted prompt with metadata."""
    system_prompt: str
    user_prompt: str
    context_length: int
    chunk_count: int


class PromptTemplates:
    """
    Strict RAG prompt templates for hallucination prevention.
    
    All prompts follow these principles:
    - Explicit instruction to use ONLY provided context
    - Clear "I don't know" directive for missing info
    - Citation requirements for traceability
    - No outside knowledge allowed
    """
    
    # Core system prompt - the foundation of hallucination prevention
    STRICT_RAG_SYSTEM_PROMPT = """You are an enterprise AI assistant that answers questions STRICTLY based on the provided document context.

## CRITICAL RULES - YOU MUST FOLLOW THESE EXACTLY:

### Rule 1: ONLY USE PROVIDED CONTEXT
- You may ONLY use information explicitly stated in the CONTEXT section below
- Do NOT use your training knowledge, even if you're confident it's correct
- Do NOT make inferences beyond what the documents directly state
- Do NOT combine information in ways the documents don't support

### Rule 2: ADMIT WHEN YOU DON'T KNOW
- If the answer is NOT in the provided context, respond with:
  "I don't have enough information in the provided documents to answer this question."
- If the answer is PARTIALLY in the context, say what you CAN answer and clearly state what information is missing
- It is BETTER to say "I don't know" than to provide potentially incorrect information

### Rule 3: CITE YOUR SOURCES
- For every factual claim, indicate which document it came from
- Use the format: [Source: filename, Page X]
- If information comes from multiple sources, cite all of them

### Rule 4: BE PRECISE
- Use the exact wording from documents when possible
- Do not paraphrase in ways that could change meaning
- If a number or date is mentioned, quote it exactly

### Rule 5: ACKNOWLEDGE LIMITATIONS
- If documents contain conflicting information, point this out
- If information seems outdated, mention the document date if available
- If a question requires information beyond the documents, say so

## RESPONSE FORMAT:
1. Provide a clear, direct answer based on the documents
2. Include source citations
3. If relevant, note any limitations or caveats

Remember: Your value comes from being ACCURATE and TRUSTWORTHY, not from always having an answer. Users trust you MORE when you admit uncertainty."""

    # User prompt template with context injection
    USER_PROMPT_TEMPLATE = """## CONTEXT FROM DOCUMENTS:
{context}

---

## USER QUESTION:
{question}

---

## YOUR TASK:
Answer the user's question using ONLY the information provided in the CONTEXT section above.
Remember: If the answer is not in the context, say "I don't have enough information in the provided documents to answer this question."
"""

    # Prompt for when no relevant context was found
    NO_CONTEXT_PROMPT = """## CONTEXT FROM DOCUMENTS:
No relevant information was found in the uploaded documents for this query.

---

## USER QUESTION:
{question}

---

## YOUR TASK:
Inform the user that no relevant information was found in the documents to answer their question.
Suggest they:
1. Rephrase their question
2. Upload additional relevant documents
3. Check if they're asking about content that exists in the uploaded documents
"""

    # Prompt for follow-up questions with conversation history
    FOLLOWUP_PROMPT_TEMPLATE = """## PREVIOUS CONVERSATION:
{history}

---

## CONTEXT FROM DOCUMENTS:
{context}

---

## FOLLOW-UP QUESTION:
{question}

---

## YOUR TASK:
Answer the follow-up question using ONLY the document context provided.
Consider the conversation history for context about what the user is asking.
If this is a new topic not covered in the documents, say so clearly.
"""

    @classmethod
    def format_rag_prompt(
        cls,
        question: str,
        context_chunks: List[str],
        conversation_history: Optional[List[dict]] = None
    ) -> FormattedPrompt:
        """
        Format a complete RAG prompt with context.
        
        Args:
            question: User's question
            context_chunks: List of relevant text chunks
            conversation_history: Optional previous Q&A pairs
            
        Returns:
            FormattedPrompt with system and user prompts
        """
        # Build context string
        if context_chunks:
            context = "\n\n---\n\n".join(context_chunks)
        else:
            context = ""
        
        # Choose appropriate template
        if not context_chunks:
            user_prompt = cls.NO_CONTEXT_PROMPT.format(question=question)
        elif conversation_history:
            history_str = cls._format_history(conversation_history)
            user_prompt = cls.FOLLOWUP_PROMPT_TEMPLATE.format(
                history=history_str,
                context=context,
                question=question
            )
        else:
            user_prompt = cls.USER_PROMPT_TEMPLATE.format(
                context=context,
                question=question
            )
        
        return FormattedPrompt(
            system_prompt=cls.STRICT_RAG_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            context_length=len(context),
            chunk_count=len(context_chunks)
        )
    
    @classmethod
    def _format_history(cls, history: List[dict]) -> str:
        """Format conversation history for prompt."""
        formatted = []
        for turn in history[-3:]:  # Last 3 turns only
            role = turn.get("role", "user")
            content = turn.get("content", "")
            formatted.append(f"{role.upper()}: {content}")
        return "\n\n".join(formatted)
    
    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the core system prompt."""
        return cls.STRICT_RAG_SYSTEM_PROMPT
    
    @classmethod
    def format_simple_prompt(cls, question: str, context: str) -> str:
        """
        Simple prompt formatting for basic usage.
        
        Returns combined system + user prompt as single string.
        """
        return f"""{cls.STRICT_RAG_SYSTEM_PROMPT}

---

{cls.USER_PROMPT_TEMPLATE.format(context=context, question=question)}"""
