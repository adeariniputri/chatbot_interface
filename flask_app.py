import re
from flask import Flask, render_template, request, jsonify
from consumer_model import autoChat
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def response_to_user():
    userText = request.args.get('msg')
    res = autoChat(userText)
    return str(res[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

