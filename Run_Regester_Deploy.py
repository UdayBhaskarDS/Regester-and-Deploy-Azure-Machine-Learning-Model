
from azureml.core import ScriptRunConfig


from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.run import Run
import shutil
import logging
import argparse
import pickle
import joblib


subscription_id = 'ad4d14ed-5af6-4288-9a19-a81cdcaf5b42'
resource_group = 'aml_RG'
workspace_name = 'aml_ws'


 # Get the run context
run = Run.get_context()
ws = Workspace(subscription_id, resource_group, workspace_name)

##To Run on curated environment
env = Environment.get(workspace=ws, name="AzureML-Tutorial")

experiment = Experiment(workspace=ws, name='Regester-Deploy')

config = ScriptRunConfig(source_directory='./', script='Regester_Deploy.py', compute_target='aml-ci')

config.run_config.environment = env

run = experiment.submit(config)

aml_url = run.get_portal_url()
print(aml_url)


