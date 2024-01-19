# Importing Libraries 
from flask import Flask , jsonify ,request

from preprocessing import remove_lines , expand_text , accented_char , clean_data , lemmatization, join_list
import pickle

# Initialization
app = Flask(__name__)

# Config data 
count_model = pickle.load(open("main/count2.pkl" , "rb"))

model = pickle.load(open("main/model2.pkl" , "rb"))

# Test route 
@app.route("/")
def home():
    return jsonify({"Response":"This is Home !"})

# Prediction route
@app.route("/predict",methods = ["POST"])
def predict():
    requested_data = request.get_data(as_text=True)

    clean_text_train = remove_lines(requested_data)

    clean_text_train = expand_text(clean_text_train)

    clean_text_train = accented_char(clean_text_train)

    clean_text_train = clean_data(clean_text_train)

    clean_text_train = lemmatization(clean_text_train)

    clean_text_train = join_list(clean_text_train)

    vector = count_model.transform([clean_text_train])

    prediction = model.predict(vector)

    if prediction[0] == 0:
        result = "Negative Sentiment"
    if prediction[0] == 1:
        result = "Neutral Sentiment"
    if prediction[0] == 2:
        result = "Positive Sentiment"

    return jsonify({"Product Review" : requested_data , "predictions_made" : result})

# Run the APP
if __name__ == "__main__":
    app.run(port=8080)
    