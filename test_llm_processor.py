#!/usr/bin/env python3
"""
Test Script for LLM Processor

This script tests the LLM processor functionality to ensure it can:
1. Connect to the OpenAI API
2. Process single transcripts
3. Handle conversation history
4. Return proper responses

Debug Info:
- Check test_llm.log for detailed logging
- All tests are timed and logged
- API key issues are reported clearly
- Errors include full stack traces
"""

import asyncio
import os
import sys
import time
import logging
import traceback
from dotenv import load_dotenv
from app.utils.llm_processor import LLMProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_llm.log")
    ]
)
logger = logging.getLogger("test_llm")

# Load environment variables
load_dotenv()

async def test_simple_query():
    """
    Test a simple query to the LLM processor.
    
    This test:
    1. Creates an LLMProcessor instance
    2. Sends a simple query
    3. Verifies the response
    
    Returns:
        dict: Result dictionary with response and metadata
        
    Debug Info:
        - Test query and response are logged
        - Processing time is measured and logged
        - Success/failure is clearly reported
    """
    processor = LLMProcessor()
    
    test_transcript = "What's the weather like today?"
    system_prompt = "You are a helpful assistant. Respond concisely."
    
    logger.info(f"Testing LLM processor with transcript: '{test_transcript}'")
    start_time = time.time()
    
    result = await processor.process_transcript(test_transcript, system_prompt)
    processing_time = time.time() - start_time
    
    if result.get("success"):
        logger.info("LLM Processor test successful!")
        logger.info(f"Response: {result['response']}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
    else:
        logger.error(f"LLM Processor test failed: {result.get('error')}")
        logger.error(f"Processing time: {processing_time:.2f} seconds")
    
    return result

async def test_conversation():
    """
    Test a conversation with the LLM processor.
    
    This test:
    1. Creates an LLMProcessor instance
    2. Sends a conversation history
    3. Verifies the response
    
    Returns:
        dict: Result dictionary with response and metadata
        
    Debug Info:
        - Conversation messages are logged
        - Processing time is measured and logged
        - Success/failure is clearly reported
    """
    processor = LLMProcessor()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, can you tell me about WebSockets?"},
    ]
    
    logger.info(f"Testing LLM processor with conversation")
    logger.info(f"Messages: {messages}")
    start_time = time.time()
    
    result = await processor.chat_with_history(messages)
    processing_time = time.time() - start_time
    
    if result.get("success"):
        logger.info("LLM Conversation test successful!")
        logger.info(f"Response: {result['response']}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
    else:
        logger.error(f"LLM Conversation test failed: {result.get('error')}")
        logger.error(f"Processing time: {processing_time:.2f} seconds")
    
    return result

async def test_error_handling():
    """
    Test error handling in the LLM processor.
    
    This test:
    1. Creates an LLMProcessor instance with an invalid model
    2. Attempts to process a transcript
    3. Verifies the error handling
    
    Returns:
        bool: True if error handling works as expected
        
    Debug Info:
        - Intentional error is logged
        - Error handling behavior is reported
    """
    logger.info("Testing error handling with invalid model")
    
    try:
        # Create processor with invalid model name
        processor = LLMProcessor(model_name="invalid-model-name")
        
        # This should fail
        result = await processor.process_transcript("Hello, world!")
        
        logger.error("Error handling test failed: Expected an exception but none was raised")
        return False
    except Exception as e:
        logger.info(f"Error handling test successful: Caught expected exception: {e}")
        return True

async def main():
    """
    Run all tests for the LLM processor.
    
    This function:
    1. Checks for API key
    2. Runs all tests
    3. Reports overall success/failure
    
    Debug Info:
        - Overall test results are summarized
        - Missing API key is clearly reported
        - Unexpected errors include stack traces
    """
    logger.info("Starting LLM processor tests")
    
    # Check if API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set in environment variables or .env file")
        logger.info("Please set the OPENAI_API_KEY environment variable and try again")
        logger.info("Example: export OPENAI_API_KEY=your-api-key-here")
        logger.info("Or create a .env file with OPENAI_API_KEY=your-api-key-here")
        return 1
    
    # Track test results
    results = {
        "simple_query": False,
        "conversation": False,
        "error_handling": False
    }
    
    # Run tests
    try:
        logger.info("=== Testing simple query ===")
        simple_result = await test_simple_query()
        results["simple_query"] = simple_result.get("success", False)
        
        logger.info("\n=== Testing conversation ===")
        conv_result = await test_conversation()
        results["conversation"] = conv_result.get("success", False)
        
        logger.info("\n=== Testing error handling ===")
        error_result = await test_error_handling()
        results["error_handling"] = error_result
        
        # Report results
        logger.info("\n=== Test Results ===")
        for test_name, success in results.items():
            status = "PASSED" if success else "FAILED"
            logger.info(f"{test_name}: {status}")
        
        # Overall result
        if all(results.values()):
            logger.info("\nAll tests PASSED!")
            return 0
        else:
            logger.warning("\nSome tests FAILED!")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error during tests: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 