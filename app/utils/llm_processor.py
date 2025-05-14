import os
import logging
import time
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("llm_processor.log")
    ]
)
logger = logging.getLogger("llm_processor")

# Load environment variables
load_dotenv()

class LLMProcessor:
    """
    A utility class for processing transcripts with LLM models using LangChain.
    
    This class handles communication with OpenAI's API through LangChain,
    processing transcripts and generating responses. It supports both
    single-turn interactions and multi-turn conversations.
    
    Debug Info:
    - Check llm_processor.log for detailed logging
    - API calls are logged with timing information
    - Errors are logged with full exception details
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the LLM processor.
        
        Args:
            model_name: The name of the OpenAI model to use. If None, will use
                        the model specified in the LLM_MODEL env variable,
                        or default to "gpt-3.5-turbo".
                        
        Raises:
            Exception: If the LLM model cannot be initialized
            
        Debug Info:
            - Check if OPENAI_API_KEY is properly set in .env
            - Verify model_name is valid and accessible with your API key
        """
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
        
        # Use model from environment or default
        if not model_name:
            model_name = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
            
        try:
            self.model = ChatOpenAI(
                model=model_name,
                temperature=0.7,
                streaming=True,
                request_timeout=60  # 60 second timeout for API calls
            )
            logger.info(f"LLM Processor initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing LLM model: {e}", exc_info=True)
            raise
    
    async def process_transcript(self, transcript: str, system_prompt: str = None) -> Dict[str, Any]:
        """
        Process a transcript with the LLM model.
        
        Args:
            transcript: The transcript text to process
            system_prompt: Optional system prompt to guide the model response
            
        Returns:
            Dictionary containing the response and metadata:
            {
                "transcript": Original transcript,
                "response": LLM generated response,
                "model": Model name used,
                "success": Boolean indicating success,
                "processing_time": Time taken to process (if successful),
                "error": Error message (if unsuccessful)
            }
            
        Debug Info:
            - Empty transcripts are logged as warnings
            - API key issues are logged as errors
            - Processing time is measured and returned
            - First 50 chars of transcript and first 100 chars of response are logged
        """
        if not transcript:
            logger.warning("Empty transcript provided to LLM processor")
            return {"response": "", "error": "Empty transcript", "success": False}
        
        if not self.api_key:
            logger.error("Cannot process transcript: OPENAI_API_KEY not set")
            return {"response": "", "error": "API key not configured", "success": False}
            
        try:
            start_time = time.time()
            # Log the full transcript
            logger.info(f"Processing full transcript: \"{transcript}\"")
            
            # Create default system prompt if none provided
            if not system_prompt:
                system_prompt = "You are a helpful assistant. Respond to the user's message concisely and helpfully."
            
            # Ensure the system prompt instructs for a one-line response
            if "one line" not in system_prompt.lower() and "single line" not in system_prompt.lower():
                system_prompt += " Provide only a single line response, no matter how complex the query."
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{transcript}")
            ])
            
            # Create chain
            chain = prompt | self.model | StrOutputParser()
            
            # Process the transcript
            response = await chain.ainvoke({"transcript": transcript})
            processing_time = time.time() - start_time
            
            # Log the full response
            logger.info(f"Full LLM Response: \"{response}\"")
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            
            return {
                "transcript": transcript,
                "response": response,
                "model": self.model.model_name,
                "success": True,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing transcript with LLM: {e}", exc_info=True)
            return {
                "transcript": transcript,
                "response": "",
                "error": str(e),
                "success": False
            }
    
    async def chat_with_history(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Process a conversation with history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
                     Example: [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                        {"role": "assistant", "content": "Hi there!"},
                        {"role": "user", "content": "How are you?"}
                     ]
            
        Returns:
            Dictionary containing the response and metadata:
            {
                "response": LLM generated response,
                "model": Model name used,
                "success": Boolean indicating success,
                "processing_time": Time taken to process (if successful),
                "error": Error message (if unsuccessful)
            }
            
        Debug Info:
            - Empty message lists are logged as warnings
            - Unsupported message roles are logged as warnings
            - Processing time is measured and returned
            - Number of messages and first 100 chars of response are logged
        """
        if not messages:
            logger.warning("Empty message list provided to LLM processor")
            return {"response": "", "error": "Empty message list", "success": False}
            
        try:
            start_time = time.time()
            logger.info(f"Processing chat with {len(messages)} messages")
            
            # Convert to LangChain message format
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
                else:
                    logger.warning(f"Unsupported message role: {msg['role']}")
            
            # Process the conversation
            response = await self.model.ainvoke(langchain_messages)
            processing_time = time.time() - start_time
            
            logger.info(f"LLM Response: {response.content[:100]}...")
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            
            return {
                "response": response.content,
                "model": self.model.model_name,
                "success": True,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing chat with LLM: {e}", exc_info=True)
            return {
                "response": "",
                "error": str(e),
                "success": False
            }


# Example usage
async def test_llm_processor():
    """
    Test function to verify the LLM processor is working.
    
    This function creates an instance of LLMProcessor and tests it with a sample query.
    It's useful for quick verification that the LLM integration is working correctly.
    
    Returns:
        Dictionary with the LLM response and metadata
    """
    processor = LLMProcessor()
    result = await processor.process_transcript(
        "Hello, can you help me understand how WebSockets work?",
        "You are a technical expert. Explain concepts clearly and concisely."
    )
    
    if result.get("success"):
        logger.info("LLM Processor test successful!")
        logger.info(f"Response: {result['response']}")
    else:
        logger.error(f"LLM Processor test failed: {result.get('error')}")
    
    return result 