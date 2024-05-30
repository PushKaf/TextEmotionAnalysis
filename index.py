from flask import Flask, render_template, request
from flask_cors import CORS
from model import Model
from dotenv import load_dotenv
import os


load_dotenv()

app = Flask(__name__)
CORS(app)

app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

# Load the model objects and load all the models initially
model = Model()
bilstm_model = model.load_dl_model("lstm_model")
logreg_model = model.load_ml_model("logreg_model")
nb_model = model.load_ml_model("nb_model")
rf_model = model.load_ml_model("rf_model")

TYPE_TO_MODEL = {"1": bilstm_model, "2": logreg_model, "3": nb_model, "4": rf_model}

@app.route("/", methods=['GET','POST'])
def index():
    result = None

    if request.method == "POST":
        # Get input if the button is pressed
        usr_input = request.form['model_input']
        model_type = request.form['model_type']
        chosen_model = TYPE_TO_MODEL[model_type]

        is_ml = True if model_type != '1' else False

        # Pass into the backend and get the result
        result = model.predict(chosen_model, usr_input, is_ml=is_ml, multiple=True)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.debug = True
    app.run()
