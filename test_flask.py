from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, World! Flask is working with Python 3.11.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)