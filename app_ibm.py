

from flask import Flask,render_template,request
import requests


# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "PeKXPtjhdefxobuoHLnasB5piVoFydbv_oddDg_8XQnW"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

#model = pickle.load(open('flight.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
  return render_template("index.html")


@app.route('/prediction',methods = ['POST'])

def predict():
  name = request.form['name']
  month = request.form['month']
  dayofmonth = request.form['dayofmonth']
  dayofweek = request.form['dayofweek']
  origin = request.form['origin']
  if(origin == "msp"):
    origin1,origin2,origin3,origin4,origin5 = 0,0,0,0,1
  if(origin == "dtw"):
    origin1,origin2,origin3,origin4,origin5 = 1,0,0,0,0
  if(origin == "jfk"):
    origin1,origin2,origin3,origin4,origin5 = 0,0,1,0,0
  if(origin == "sea"):
    origin1,origin2,origin3,origin4,origin5 = 0,1,0,0,0
  if(origin == "atl"):
    origin1,origin2,origin3,origin4,origin5 = 0,0,0,1,0

  destination = request.form['destination']
  if(destination == "msp"):
    destination1,destination2,destination3,destination4,destination5 = 0,0,0,0,1
  if(destination == "dtw"):
    destination1,destination2,destination3,destination4,destination5 = 1,0,0,0,0
  if(destination == "jfk"):
    destination1,destination2,destination3,destination4,destination5 = 0,0,1,0,0
  if(destination == "sea"):
    destination1,destination2,destination3,destination4,destination5 = 0,1,0,0,0
  if(destination == "atl"):
    destination1,destination2,destination3,destination4,destination5 = 0,0,0,1,0
  dept = request.form['dept']
  arrtime = request.form['arrtime']
  actdept = request.form['actdept']
  dept15 = int(dept) - int(actdept)
  total = [[name,month,dayofmonth,dayofweek,origin1,origin2,origin3,origin4,origin5,destination1,destination2,destination3,destination4,destination5,int(arrtime),int(dept15)]]
 # y_pred = model.predict(total)
  # NOTE: manually define and pass the array(s) of values to be scored in the next line
  payload_scoring = {"input_data": [{"fields": ["name","month","dayofmonth","dayofweek","origin1","origin2","origin3","origin4","origin5","destination1","destination2","destination3","destination4","destination5","scheduleddeparturetime","actualdeparturetime"],"values":total}]}

  response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/a2dae41a-a25e-4586-a629-03d22be0208f/predictions?version=2022-06-02', json=payload_scoring,
   headers={'Authorization': 'Bearer ' + mltoken})
  print("Scoring response")
  #print(response_scoring.json())
  pred=response_scoring.json()
  output=pred['predictions'][0]['values'][0][0]
  print(output)

 # print(y_pred)

  if(output==0):
      ans ="The Flight will be on time"
  else:
      ans ="The Flight will be delayed"
  print(ans)    
  return render_template("index.html",showcase= ans)

if __name__=='__main__':
    app.run(debug=False)