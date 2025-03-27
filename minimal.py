"""
Minimal app for testing
"""

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({"message": "Hello, world!"})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    return jsonify({
        "response": f"Echo: {message}",
        "thought_process": []
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)