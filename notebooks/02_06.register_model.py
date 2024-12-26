# Databricks notebook source
# MAGIC %pip install -r requirements.in

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import json

import mlflow
import yaml
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
client = MlflowClient()

# COMMAND ----------
with open("../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

catalog_name = config.get("catalog_name")
schema_name = config.get("schema_name")

model_name = f"{catalog_name}.{schema_name}.basic_model"

# COMMAND ----------

run_id = mlflow.search_runs(
    experiment_names=["/Shared/hotel-cancellations-basic"],
    filter_string="tags.branch='02_04'",
).run_id[0]

model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model",
    name=model_name,
    tags={"git_sha": "51v63531711eaa139"},
)

# COMMAND ----------
with open("model_version.json", "w") as json_file:
    json.dump(model_version.__dict__, json_file, indent=4)

# COMMAND ----------
model_version_alias = "the_best_model"
client.set_registered_model_alias(model_name, model_version_alias, "1")

model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------
client.get_model_version_by_alias(model_name, model_version_alias)


# COMMAND ----------
# import mlflow
#
# client = mlflow.tracking.MlflowClient()
# schema = "hotel_cancellation"
#
# src_model_name = "acc.{schema}.basic_model"
# src_model_version = "1"
# src_model_uri = f"models:/{src_model_name}@{src_model_version}"
# dst_model_name = "prd.{schema}.basic_model"
#
# copied_model_version = client.copy_model_version(src_model_uri, dst_model_name)
