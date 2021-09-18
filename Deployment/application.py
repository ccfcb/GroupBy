from flask import Flask, request, render_template 
import numpy as np
import pandas as pd
from models import *

application = app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')
        
@app.route('/results',methods=['POST'])
def results():
    
    #customer = request.form.values()
    customer = request.form.get('CustomerId')
    
    # Probability of conversion prediction
    output = predict(customer)
    
    # Classification
    customer_class = classify(customer)
    
    # Recommendations
    recommendations = recommend(customer)
    
    return render_template('outputs.html', customer_text=customer, 
           prediction_text=output, 
           classification_text=customer_class,
           tables=[recommendations.to_html(header=True, index=False)])
    
if __name__ == "__main__":
    app.run(host='0.0.0.0')
   
