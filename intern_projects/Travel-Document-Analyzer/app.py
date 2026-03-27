from flask import Flask
from routes.api import api_blueprint
import os

# Replace with your actual path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\travel-doc-analyzer\travel-document-analyzer-851e9a7aae11.json"

app = Flask(__name__)
app.register_blueprint(api_blueprint, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)