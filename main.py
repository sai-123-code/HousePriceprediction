import pickle
from flask import Flask,url_for ,render_template, request
# from flask_cors import cross_origin
# from src import config

app=Flask(__name__)


@app.route("/")
# @cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict",methods=['POST','GET'])
# @cross_origin()
def pred():
    if request.method=='POST':
        # try:
        CRIM=float(request.form.get('crime'))
        ZN = float(request.form['zn'])
        INDUS = float(request.form['indus'])
        CHAS = float(request.form['chas'])
        NOX = float(request.form['nox'])
        RM = float(request.form['rm'])
        AGE = float(request.form['age'])
        DIS = float(request.form['dis'])
        RAD = float(request.form['rad'])
        PTRATIO = float(request.form['pt-ratio'])
        B = float(request.form['b'])
        LSTAT = float(request.form['ls'])

        x_predict=[[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,PTRATIO,B,LSTAT]]
        filename='final_model.pkl'
            # model=pickle.load(open(config.FINAL_MODEL,'rb'))
        model=pickle.load(open(filename,'rb'))
        prediction=model.predict(x_predict)
        return render_template('result.html', prediction=prediction)
        # except Exception as e:
        #     return e
    #
    # else:
    #     return render_template('index.html')


if __name__== "__main__":
    app.run(debug=True)

