import os
import logging
from flask import Flask, render_template, request, jsonify, session

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
            
        # Simple response for testing
        return jsonify({
            "response": f"Echo: {message}",
            "thought_process": []
        })
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return jsonify({"error": f"Error processing message: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)