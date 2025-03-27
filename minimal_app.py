from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Flask App</title>
        <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container mt-5">
            <div class="row">
                <div class="col-md-8 offset-md-2">
                    <div class="card">
                        <div class="card-header">
                            <h2>Business Information Extraction</h2>
                        </div>
                        <div class="card-body">
                            <h3>App is running successfully!</h3>
                            <p>This is a minimal test of the Flask application</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """)