# Databricks notebook source
import time
from datetime import datetime, timezone

import yaml
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

workspace = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

with open("../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

catalog_name = config.get("catalog_name")
schema_name = config.get("schema_name")
pipeline_id = config.get("pipeline_id")


# COMMAND ----------

spark.sql(
    f"CREATE TABLE {catalog_name}.{schema_name}.extra_train_set "
    f"LIKE {catalog_name}.{schema_name}.extra_set;"
)

spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.extra_train_set "
    "ADD COLUMN update_timestamp_utc TIMESTAMP;"
)

# COMMAND ----------
time_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

spark.sql(
    f"INSERT INTO {catalog_name}.{schema_name}.extra_train_set "
    f"SELECT *, '{time_now}' as update_timestamp_utc "
    f"FROM {catalog_name}.{schema_name}.extra_set where "
    "market_segment_type='Aviation';"
)

# COMMAND ----------
affected_rows = (
    spark.sql(
        f"""
    WITH max_timestamp AS (
        SELECT MAX(update_timestamp_utc) AS max_update_timestamp
        FROM {catalog_name}.{schema_name}.train_set
    )
    INSERT INTO {catalog_name}.{schema_name}.train_set
    SELECT *
    FROM {catalog_name}.{schema_name}.extra_train_set
    WHERE update_timestamp_utc > (SELECT max_update_timestamp FROM max_timestamp)
"""
    )
    .first()
    .num_affected_rows
)


# COMMAND ----------
# write into feature table; update online table
if affected_rows > 0:
    spark.sql(
        f"""
        WITH max_timestamp AS (
            SELECT MAX(update_timestamp_utc) AS max_update_timestamp
            FROM {catalog_name}.{schema_name}.train_set
        )
        INSERT INTO {catalog_name}.{schema_name}.hotel_features
        SELECT Booking_ID, repeated, P_C, P_not_C 
        FROM {catalog_name}.{schema_name}.train_set
        WHERE update_timestamp_utc == (SELECT max_update_timestamp FROM max_timestamp)
"""
    )
# COMMAND ----------

update_response = workspace.pipelines.start_update(
    pipeline_id=pipeline_id, full_refresh=False
)

while True:
    update_info = workspace.pipelines.get_update(
        pipeline_id=pipeline_id, update_id=update_response.update_id
    )
    state = update_info.update.state.value
    if state == "COMPLETED":
        success = 1
        break
    elif state in ["FAILED", "CANCELED"]:
        success = 0
        break
    elif state == "WAITING_FOR_RESOURCES":
        print("Pipeline is waiting for resources.")
    else:
        print(f"Pipeline is in {state} state.")
    time.sleep(30)

print(success)
# COMMAND ----------
last_version_train_set = spark.sql(
    f"DESCRIBE HISTORY {catalog_name}.{schema_name}.train_set"
).first()

last_version_train_set.version

# COMMAND ----------
spark.sql(
    f"INSERT INTO {catalog_name}.{schema_name}.extra_train_set "
    f"SELECT *, '{time_now}' as update_timestamp_utc "
    f"FROM {catalog_name}.{schema_name}.extra_set where "
    "market_segment_type='Complimentary';"
)
