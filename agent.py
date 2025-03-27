import os
import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.memory import ChatMessageHistory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_manager, create_memory_store_manager
from langmem.utils import NamespaceTemplate

from tools import get_tools

# Configure logging
logger = logging.getLogger(__name__)

# Define memory schemas
class UserProfile(BaseModel):
    """Represents user-specific information."""
    name: str | None = Field(None, description="User's name")
    timezone: str | None = Field(None, description="User's timezone")
    preferences: dict[str, str] = Field(default_factory=dict, description="Key-value preferences")

class SemanticTriple(BaseModel):
    """Stores a subject-predicate-object fact."""
    subject: str
    predicate: str
    object_: str = Field(alias="object", description="The object of the triple")

def create_agent():
    """Create a React agent with memory capabilities.
    
    Returns:
        Tuple containing the agent executor, memory store, and semantic memory manager
    """
    try:
        # Get OpenAI API key from environment variables
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize the language model - using the latest available model
        llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0,
            api_key=openai_api_key
        )
        
        # Initialize memory store with embedding capability
        store = InMemoryStore(
            index={
                "dims": 1536,  # Dimension for text-embedding-3-small
                "embed": "openai:text-embedding-3-small"  # Specify the embedding model
            }
        )
        
        # Set up namespaces for different memory types
        user_profile_ns = NamespaceTemplate(("profiles", "{user_id}"))
        semantic_memory_ns = NamespaceTemplate(("semantic_memories", "{user_id}"))
        
        # Memory manager for extracting and storing user profile info
        profile_manager = create_memory_store_manager(
            "gpt-4-turbo",  # Model as positional arg
            schemas=[UserProfile],
            instructions="Analyze the conversation and extract/update the user's profile information.",
            namespace=user_profile_ns,
            store=store,
            enable_inserts=True,
            enable_deletes=True  # 'enable_updates' was renamed to 'enable_deletes'
        )
        
        # Memory manager for extracting semantic triples
        semantic_manager = create_memory_store_manager(
            "gpt-4-turbo",  # Model as positional arg
            schemas=[SemanticTriple],
            instructions="Extract factual relationships as subject-predicate-object triples.",
            namespace=semantic_memory_ns,
            store=store,
            enable_inserts=True,
            enable_deletes=True  # 'enable_updates' was renamed to 'enable_deletes'
        )
        
        # Get tools for the agent
        tools = get_tools()
        
        # Create a template that includes all required variables
        template = """You are a helpful assistant with access to the internet and a memory system. 
You can search the web and remember information from previous conversations.
Always provide informative and truthful responses based on information you can access.
If you don't know something, you can search for it or be honest about not knowing.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Chat History:
{chat_history}

Question: {input}
{agent_scratchpad}
"""
        
        # Create prompt with all required variables in a single template
        prompt = PromptTemplate.from_template(template)
        
        # Initialize chat memory history
        chat_history = ChatMessageHistory()
        logger.info("Using standard ChatMessageHistory for agent memory")
            
        # Create custom memory using the history
        class CustomChatMemory(BaseChatMemory):
            def __init__(self, chat_history, memory_key="chat_history", return_messages=True):
                super().__init__(memory_key=memory_key, return_messages=return_messages)
                self.chat_history = chat_history
                
            @property
            def memory_variables(self) -> List[str]:
                return [self.memory_key]
                
            def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                return {self.memory_key: self.chat_history.messages}
                
            def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
                # Save user message
                if "input" in inputs:
                    self.chat_history.add_user_message(inputs["input"])
                # Save AI message
                if outputs and isinstance(outputs, dict) and "output" in outputs:
                    self.chat_history.add_ai_message(outputs["output"])
                    
            def clear(self) -> None:
                """Clear memory contents."""
                self.chat_history.clear()
                    
        # Use the custom memory handler
        memory = CustomChatMemory(chat_history, memory_key="chat_history", return_messages=True)
        
        # Create the React agent with proper formatting for tool messages
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
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
        
        logger.info("Agent created successfully with memory management capabilities")
        
        # Return the agent executor, store, and semantic manager for reflection
        return agent_executor, store, semantic_manager
    
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise
