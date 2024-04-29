from flask import *
from flask_cors import CORS
from numpy import *
from pickle import *
import numpy as np
import re
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from flask_pymongo import pymongo
import bcrypt
import time
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import hashlib

train = pd.read_csv('./Training.csv')
test = pd.read_csv('./Testing.csv')

train = train.drop(["Unnamed: 133"], axis=1)

label_encoder = LabelEncoder()
train['prognosis'] = label_encoder.fit_transform(train['prognosis'])
test['prognosis'] = label_encoder.transform(test['prognosis'])

X_train = train.drop(['prognosis'], axis=1)
y_train = train['prognosis']
X_test = test.drop(['prognosis'], axis=1)
y_test = test['prognosis']

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier()
}

symptoms = list(X_train.columns)

app = Flask(__name__)
CORS(app)
app.secret_key = "this is very confidential"

app.config['BASE_URL'] = "http://localhost:5000/"

@app.route("/home")         # Home page
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("About_pag.html")

@app.route("/contact")
def contact():
    return render_template("Contact.html")

@app.route("/doctor")
def doctors():
    return render_template("doctor.html")

@app.route("/testimonial")
def testimonials():
    return render_template("testimonial.html")

@app.route("/treatment")
def treatments():
    return render_template("treatment.html")

uri = "mongodb+srv://ninadsugandhi:disease@cluster0.2sqe1dd.mongodb.net/?tls=truee&tlsInsecure=True&retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client.diseasepredictordb
collection = db.userdb

@app.route("/home", methods=["GET", "POST"])  # Combined login and registration route
def login_or_register():
    if request.method == "POST":
        if "login" in request.form:
            # Handle login
            email = request.form["email"]
            password = request.form["password"]

            # Query MongoDB for user credentials
            user = collection.find_one({"email": email})

            if user and user["password"] == password:
                session["Loggedin"] = True
                session["id"] = str(user["_id"])
                session["email"] = email
                return redirect("/model")
            else:
                msg = "Incorrect Email / Password"
                return render_template("index.html", msg=msg)

        elif "register" in request.form:
            # Handle registration
            name = request.form["name"]
            email = request.form["email"]
            password = request.form["password"]

            # Check if user already exists
            if collection.find_one({"email": email}):
                msg = "Account already exists!"
            else:
                # Insert user into MongoDB
                collection.insert_one({"name": name, "email": email, "password": password})
                msg = "Account created successfully!"

            return render_template("index.html", msg=msg)

    return render_template("index.html", msg=None)

@app.route("/model", methods=["POST", "GET"])  
def predict():
    if "Loggedin" in session:
        predictions = None
        if request.method == "POST":
            selected_symptoms = request.form.getlist('symptoms')
            if len(selected_symptoms) < 3:
                return "Please select at least three symptoms before predicting."
            else:
                user_input = {}
                for symptom in symptoms:
                    user_input[symptom] = 1 if symptom in selected_symptoms else 0
                input_df = pd.DataFrame([user_input])
                predictions = {}
                for model_name, model in models.items():
                    # Fit the model with training data
                    model.fit(X_train, y_train)
                    proba = model.predict_proba(input_df)[0]
                    pred = np.argmax(proba)
                    pred_name = label_encoder.inverse_transform([pred])[0]
                    predictions[model_name] = pred_name
        return render_template("model.html", symptoms=symptoms, predictions=predictions)
    else:
        return redirect("/home")
    

if __name__ == "__main__":
    app.run(debug=True)