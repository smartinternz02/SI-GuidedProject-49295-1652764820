#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
get_ipython().system('pip install imblearn')
import imblearn
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from matplotlib import pyplot as plt


# In[3]:


import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
client_850d1fb7aa8543d4b9e87d5770a697d6 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='KOqJVMvn3LwJ5yrGH3yI-6zJo1FMfhaqcVxkunOoINjG',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

body = client_850d1fb7aa8543d4b9e87d5770a697d6.get_object(Bucket='flightdelayprediction-donotdelete-pr-pn4cj2wgyfysdo',Key='flightdata.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset = pd.read_csv(body)
dataset.head()


# In[4]:


dataset.info()


# In[5]:


dataset.describe()


# In[6]:


dataset.isnull().sum()


# In[7]:


dataset["DEST"].unique()


# In[8]:


plt.scatter(dataset['ARR_DELAY'],dataset['ARR_DEL15'])
plt.xlabel('ARR_DELAY')
plt.ylabel('ARR_DEL15')
plt.title('Scatter plot between ARR_DELAY and ARR_DEL15')


# In[9]:


#sns.catplot(x='ARR_DELAY',y='ARR_DEL15', data = flightdata)
sns.heatmap(dataset.corr())


# In[10]:


dataset = dataset.drop('Unnamed: 25', axis=1)
dataset.isnull().sum()


# In[11]:


dataset = dataset [["FL_NUM","MONTH","DAY_OF_MONTH","DAY_OF_WEEK","ORIGIN","DEST","CRS_ARR_TIME","DEP_DEL15","ARR_DEL15"]]

dataset.isnull().sum()


# In[12]:


dataset = dataset.fillna({'ARR_DEL15': 1})
dataset = dataset.fillna({'DEP_DEL15': 0})
dataset.iloc[177:185]


# In[13]:


import math

for index, row in dataset.iterrows():
  dataset.loc[index,'CRS_ARR_TIME'] = math.floor(row['CRS_ARR_TIME']/100)
dataset.head()


# In[14]:


#from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
#label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
#dataset['DEST']= label_encoder.fit_transform(dataset['DEST']) 
#dataset['DEST'].unique()
#dataset['ORIGIN']= label_encoder.fit_transform(dataset['ORIGIN']) 
#dataset['ORIGIN'].unique()
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
dataset['DEST']=le.fit_transform(dataset['DEST'])
dataset['ORIGIN']=le.fit_transform(dataset['ORIGIN'])


# In[15]:


dataset.head(5)


# In[16]:


x = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8:9].values
x.shape


# In[17]:


y


# In[18]:


from sklearn.preprocessing import OneHotEncoder
oh=OneHotEncoder()
z=oh.fit_transform(x[:,4:5]).toarray()
t=oh.fit_transform(x[:,5:6]).toarray()


# In[19]:


#one = OneHotEncoder()
#z = one.fit_transform(x[:,0:10]).toarray()
t


# In[20]:


z


# In[21]:


x=np.delete(x,[4,5],axis=1) 
x.shape


# In[22]:


x=np.concatenate((t,z,x),axis = 1)
x.shape


# In[23]:


dataset = pd.get_dummies(dataset, columns=['ORIGIN','DEST'])
dataset.head()


# In[24]:


#x = dataset.iloc[:,0:16].values
y = dataset.iloc[:,5:6].values
#y


# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[26]:


x_test.shape


# In[27]:


x_train.shape


# In[28]:


y_test.shape


# In[29]:


y_train.shape


# In[30]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[ ]:





# In[ ]:





# In[31]:


# balancing the dataset using SMOTE technique
from imblearn.over_sampling import SMOTE
smote = SMOTE()


# In[32]:


x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)


# In[33]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state = 0)
classifier.fit(x_train_smote,y_train_smote)


# In[34]:


decisiontree = classifier.predict(x_test)


# In[35]:


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,decisiontree)
acc


# In[36]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,decisiontree)


# In[37]:


cm


# In[38]:


import pickle
pickle.dump(classifier,open("flight.pkl","wb"))


# In[39]:


pwd


# # Deployment

# dallas https://us-south.ml.cloud.ibm.com

# In[40]:


get_ipython().system('pip install -U ibm-watson-machine-learning')


# In[41]:


from ibm_watson_machine_learning import APIClient
import json
import numpy as np


# # Authenticate and Set Space

# In[42]:


wml_credentials={
    "apikey":"PeKXPtjhdefxobuoHLnasB5piVoFydbv_oddDg_8XQnW",
    "url":"https://us-south.ml.cloud.ibm.com"
}


# In[43]:


wml_client=APIClient(wml_credentials)
wml_client.spaces.list()


# In[44]:


SPACE_ID="fe14ac3f-644b-499d-992c-980a31348ba4"


# In[45]:


wml_client.set.default_space(SPACE_ID)


# In[46]:


wml_client.software_specifications.list()


# # save and deploy the model

# In[47]:


import sklearn
sklearn.__version__


# In[48]:


MODEL_NAME="flightdelaymodel"
DEPLOYMENT_NAME="flightdelay_model"
DEMO_MODEL=classifier


# In[49]:


software_spec_uid=wml_client.software_specifications.get_id_by_name('runtime-22.1-py3.9')


# In[50]:


model_props={
    wml_client.repository.ModelMetaNames.NAME:MODEL_NAME,
    wml_client.repository.ModelMetaNames.TYPE:'scikit-learn_1.0',
    wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:software_spec_uid
    
}


# In[51]:


model_details= wml_client.repository.store_model(model=DEMO_MODEL,meta_props=model_props,training_data=x_train,training_target=y_train)


# In[52]:


model_details


# In[53]:


model_id=wml_client.repository.get_model_id(model_details)


# In[54]:


deployment_props={
    wml_client.deployments.ConfigurationMetaNames.NAME:DEPLOYMENT_NAME,
    wml_client.deployments.ConfigurationMetaNames.ONLINE:{}
}


# In[55]:


deployment= wml_client.deployments.create(
    artifact_uid=model_id,
    meta_props=deployment_props
)


# In[ ]:




