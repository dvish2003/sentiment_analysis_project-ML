from flask import Flask,render_template,request,redirect
from helper import preprocessing,vectorizer,get_prediction
app = Flask(__name__)

data = dict()
reviews = []
positive_reviews = 0
negative_reviews = 0

@app.route('/')
def index():
    data['reviews'] = reviews
    data['positive_reviews'] = positive_reviews
    data['negative_reviews'] = negative_reviews
    return render_template('index.html', data=data)

@app.route("/",methods = ['post'])
def my_post():
    text = request.form['text']
    preprocess_text = preprocessing(text)
    vectorized = vectorizer(preprocess_text)
    prediction = get_prediction(vectorized)
     
    if prediction == 'negative':
        global negative_reviews 
        negative_reviews += 1
    else:
        global positive_reviews
        positive_reviews += 1
    reviews.insert(0, text)
    return redirect(request.url)
        

if __name__ == "__main__":
    app.run()