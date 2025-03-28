from flask import Flask, render_template_string

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Business Information Extraction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .upload-form {
            margin-top: 20px;
        }
        .btn {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .btn:hover {
            background-color: #2980b9;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Business Information Extraction</h1>
        <p>This application extracts business owner information from Excel files using advanced techniques:</p>
        <ul>
            <li>Multi-query generation for comprehensive search</li>
            <li>Parallel processing with maximum concurrency</li>
            <li>Advanced content analysis with React-style reasoning</li>
            <li>Dynamic search expansion based on confidence thresholds</li>
        </ul>
        <div class="upload-form">
            <h2>Upload Excel File</h2>
            <p>Select an Excel file containing business information to process:</p>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".xlsx,.xls">
                <button type="submit" class="btn">Process File</button>
            </form>
        </div>
        <div class="status-section" id="status-section" style="display: none;">
            <h2>Processing Status</h2>
            <div id="status-content">
                <p>Processing your file...</p>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)