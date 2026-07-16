from flask import Flask, request, render_template, jsonify
import joblib
from chatbot import get_ai_response

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():

    # Get message from HTML form
    message = request.form["message"]

    # Convert text into numbers
    data = vectorizer.transform([message])

    # Prediction
    prediction = model.predict(data)
    probability = model.predict_proba(data)

    print("Message:", message)
    print("Prediction:", prediction)
    print("Probability:", probability)

    # Result
    if prediction[0] == 1:
        result = "🚨 Spam Email"
    else:
        result = "✅ Not Spam Email"

    return render_template("index.html", prediction=result)


# ===========================
# AI Cyber Assistant Route
# ===========================
@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    user_message = data.get("message")

    reply = get_ai_response(user_message)

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)