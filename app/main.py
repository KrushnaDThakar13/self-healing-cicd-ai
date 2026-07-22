from flask import Flask, jsonify
from prometheus_flask_exporter import PrometheusMetrics
import random
import time

app = Flask(__name__)
metrics = PrometheusMetrics(app)

@app.route("/")
def home():
    return jsonify({"message": "Payment service is running"}), 200

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/pay")
def pay():
    if random.random() < 0.05:
        return jsonify({"error": "Payment failed"}), 500
    time.sleep(0.1)
    return jsonify({"status": "success", "transaction_id": random.randint(1000, 9999)}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)