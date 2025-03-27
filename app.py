import os
import logging
import uuid
from flask import Flask, render_template, request, jsonify, session
from agent import create_agent
from langmem import ReflectionExecutor

# Try to import business routes
try:
    from business_routes import business_bp
    BUSINESS_ROUTES_AVAILABLE = True
except ImportError:
    BUSINESS_ROUTES_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

# Initialize agent and reflection executor for background memory processing
agent = None
reflection_executor = None

def initialize_agent():
    global agent, reflection_executor
    try:
        logger.info("Initializing agent...")
        # Initialize agent and get memory components
        # We'll modify create_agent() to return the agent, store, and memory manager
        agent_info = create_agent()
        if isinstance(agent_info, tuple) and len(agent_info) == 3:
            agent, store, manager = agent_info
            # Create ReflectionExecutor
            # The namespace is a positional argument
            namespace = ("memory_reflection", "{user_id}")  # Dynamic namespace using user_id
            reflection_executor = ReflectionExecutor(manager, namespace, store=store)
        else:
            agent = agent_info
            reflection_executor = None
            logger.warning("Memory reflection not enabled - missing store or manager")
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing agent: {str(e)}")
        raise

# Will initialize the agent on first request automatically
with app.app_context():
    try:
        initialize_agent()
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")

@app.route('/')
def index():
    # Set a unique user ID for memory namespacing if not already set
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global agent, reflection_executor
    
    # Reinitialize if needed
    if agent is None:
        try:
            initialize_agent()
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            return jsonify({"error": f"Agent initialization failed: {str(e)}"}), 500
    
    try:
        # Get or create user_id for memory namespacing
        user_id = session.get('user_id', str(uuid.uuid4()))
        if 'user_id' not in session:
            session['user_id'] = user_id
            
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        logger.debug(f"Received message: {message}")
        
        # Invoke agent with user_id for namespace resolution
        config = {"configurable": {"user_id": user_id}}
        # If agent is an executor (not a tuple), use it directly
        if not isinstance(agent, tuple):
            response = agent.invoke({"input": message}, config=config)
        else:
            # If agent is a tuple, agent_executor is the first element
            agent_executor = agent[0]
            response = agent_executor.invoke({"input": message}, config=config)
        
        logger.debug(f"Agent response: {response}")
        
        # Store message history for background processing
        if reflection_executor is not None:
            # Queue background memory processing
            # This would normally include the agent's memory managers
            # that were created in the agent.py file
            messages = [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response["output"]}
            ]
            reflection_executor.submit(
                {"messages": messages}, 
                config=config,
                after_seconds=1  # Process after a delay
            )
        
        return jsonify({
            "response": response["output"],
            "thought_process": response.get("intermediate_steps", [])
        })
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return jsonify({"error": f"Error processing message: {str(e)}"}), 500
