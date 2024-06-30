from flask import Flask,render_template,request
import pandas as pd
import sys
from source.exception import UserException
from source.commons import load_object

app = Flask(__name__)

preprocessor_path = 'elements\preprocessor.pkl'
model_path = 'elements\model.pkl'

preprocessor = load_object(preprocessor_path)
model = load_object(model_path)

@app.route('/')
def homepage():
    return render_template('emp.html')

@app.route('/predict',methods=['POST'])
def start_prediction():
    try:
        a = request.form.get('gender')
        b = request.form.get('years')
        c = request.form.get('work-life-balance')
        d = request.form.get('job-satisfaction')
        e = request.form.get('performance-rating')
        f = request.form.get('promotions')
        g = request.form.get('overtime')
        h = request.form.get('distance')
        i = request.form.get('education-level')
        j = request.form.get('marital-status')
        k = request.form.get('dependents')
        l = request.form.get('job-level')
        m = request.form.get('company-size')
        n = request.form.get('remote')
        o = request.form.get('company-reputation')
        p = request.form.get('employee-recognition')
        
        columns = ['gender','years at company','work-life balance','job satisfaction','performance rating','number of promotions','overtime','distance from home','education level','marital status',
                   'number of dependents','job level','company size','remote work','company reputation','employee recognition']
        x = pd.DataFrame([[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p]],columns=columns)
        
        x = preprocessor.transform(x)
        pred = model.predict(x)
        msg = 'Employee is likely to stay' if pred == 1 else 'Employee is likely to leave'
        
        return render_template('emp.html',text=msg)
    except Exception as e:
        raise UserException(e,sys)
    
if __name__ == "__main__":
    app.run(debug=True)