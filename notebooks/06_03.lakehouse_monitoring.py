# Databricks notebook source
import yaml
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)

workspace = WorkspaceClient()

with open("../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

catalog_name = config.get("catalog_name")
schema_name = config.get("schema_name")


# COMMAND ----------

monitoring_table = f"{catalog_name}.{schema_name}.model_monitoring"

workspace.quality_monitors.create(
    table_name=monitoring_table,
    assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
    output_schema_name=f"{catalog_name}.{schema_name}",
    inference_log=MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
        prediction_col="prediction",
        timestamp_col="timestamp",
        granularities=["30 minutes"],
        model_id_col="model_name",
        label_col="booking_status",
    ),
)

spark.sql(
    f"ALTER TABLE {monitoring_table} "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)
