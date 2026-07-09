from flask import Flask, jsonify, request
import os

app = Flask(__name__)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Simple server running"})

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"test": "success"})

if __name__ == '__main__':
    print("Starting simple server on port 5002...")
    app.run(host='0.0.0.0', port=5002, debug=False)