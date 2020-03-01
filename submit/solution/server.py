from flask import Flask, jsonify, request
from src.two_stage_predictor import TwoStagePredictor

PREDICTOR = TwoStagePredictor(assets_root="assets/")

app = Flask(__name__)


@app.route("/ready")
def ready():
    return "OK"


@app.route("/recommend", methods=["POST"])
def recommend():
    r = request.json
    result = PREDICTOR.predict(r, k=30)
    return jsonify({"recommended_products": result})


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host="0.0.0.0", debug=False, port=8000)
