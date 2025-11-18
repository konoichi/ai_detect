import os
from flask import Flask, send_from_directory, jsonify
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "")
PORT = int(os.getenv("PORT", 5050))

app = Flask(__name__, static_folder="static")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)

@app.route("/config.json")
def config():
    return jsonify({"API_BASE_URL": API_BASE_URL})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
