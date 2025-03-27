import os
import logging
from flask import Flask, render_template, request, jsonify
from agent import create_agent

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

# Initialize agent
agent = None

@app.before_first_request
def initialize_agent():
    global agent
    try:
        logger.info("Initializing agent...")
        agent = create_agent()
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing agent: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global agent
    if agent is None:
        try:
            agent = create_agent()
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            return jsonify({"error": f"Agent initialization failed: {str(e)}"}), 500
    
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        logger.debug(f"Received message: {message}")
        response = agent.invoke({"input": message})
        logger.debug(f"Agent response: {response}")
        
        return jsonify({
            "response": response["output"],
            "thought_process": response.get("intermediate_steps", [])
        })
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return jsonify({"error": f"Error processing message: {str(e)}"}), 500
