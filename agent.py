import os
import logging
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langmem import integrate_memory

from tools import get_tools

# Configure logging
logger = logging.getLogger(__name__)

def create_agent() -> AgentExecutor:
    """Create a React agent with langmem memory integration."""
    try:
        # Get OpenAI API key from environment variables
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize the language model
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0,
            api_key=openai_api_key
        )
        
        # Get tools for the agent
        tools = get_tools()
        
        # Create prompt for the agent
        system_message = """You are a helpful assistant with access to the internet and a memory system. 
You can search the web and remember information from previous conversations.
Always provide informative and truthful responses based on information you can access.
If you don't know something, you can search for it or be honest about not knowing.
"""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Initialize memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Create the React agent
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
        )
        
        # Integrate langmem memory
        agent_executor_with_memory = integrate_memory(agent_executor)
        
        logger.info("Agent created successfully with langmem memory integration")
        return agent_executor_with_memory
    
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise
