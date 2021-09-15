# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 18:38:43 2021

@author: Bhaskar
"""

from azureml.core.model import Model
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required#
from azureml.core import Workspace, Dataset,Experiment,Environment
from azureml.core.run import Run
#from sklearn.externals import joblib
import joblib

import numpy as np
import matplotlib as pyplot
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

subscription_id = 'ad4d14ed-5af6-4288-9a19-a81cdcaf5b42'
resource_group = 'aml_RG'
workspace_name = 'aml_ws'
run = Run.get_context()
workspace = Workspace(subscription_id, resource_group, workspace_name)
env = Environment.get(workspace=workspace, name="AzureML-Tutorial")
datastore = workspace.get_default_datastore()

experiment = Experiment(workspace=workspace, name='Model_Training_Deploy')
input_dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, './features.csv')])
output_dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, './labels.csv')])
  
'''model = Model.register(workspace =workspace ,model_name='Diabetes_Classification1', model_path="./outputs/Modelfile_for_OP/BestModel.pkl",)
print(model.name, model.id, model.version, sep='\t')'''
create_new_model = False
registered_model_name = 'Diabetes_Classification2'
model_version = 1
#Retrieve the path to the model file using the model name
print(Model.get_model_path(model_name=registered_model_name))
model_path = Model.get_model_path(model_name = registered_model_name)

model_list = Model.list(workspace = workspace)
model_count = len(model_list)
#Print all the models
for i in range(len(model_list)):
    print("{0}. {1}\n".format(i+1, model_list[i]))
    # Limit only latest model
    if i == 0:
        break
#Connect to model
if not create_new_model:
    print("Connecting to model...")
    model = Model(workspace=workspace, name=registered_model_name, version=model_version)
    print(model)
else:
    print('Skipping...')
    

    
#Load the model
#model = joblib.load(model_path)

env = Environment.get(workspace=workspace, name="AzureML-Tutorial")


service_name = 'my-sklearn-service'

service = Model.deploy(workspace, service_name, [model], overwrite=True)
service.wait_for_deployment(show_output=True)
workspace.webservices[service_name].get_logs()


import json

input_dataset = input_dataset.to_pandas_dataframe()
input_payload = json.dumps({
    'data': input_dataset.values[0:2].tolist(),
    'method': 'predict_proba'  # If you have a classification model, you can get probabilities by changing this to 'predict_proba'.
})

output = service.run(input_payload)

print(output)
'''#For Custom environment
service_name = 'my-custom-env-service'

inference_config = InferenceConfig(entry_script='score.py', environment=env)
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(workspace=workspace,
                       name=service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=aci_config,
                       overwrite=True)
service.wait_for_deployment(show_output=True)
##Call servise
import json
    
input_payload = json.dumps({
    'data': input_dataset[0:2].tolist()
})

output = service.run(input_payload)

print(output)'''