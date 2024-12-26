import yaml
import argparse
from pyspark.sql import SparkSession
import yaml
from databricks.sdk import WorkspaceClient
import time

workspace = WorkspaceClient()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path",
        action="store",
        default=None,
        type=str,
        required=True,
    )

    args = parser.parse_args()
    root_path = args.root_path

    with open(f"/Workspace/{root_path}/files/project_config.yml", "r") as file:
        config = yaml.safe_load(file)

spark = SparkSession.builder.getOrCreate()

catalog_name = config.get("catalog_name")
schema_name = config.get("schema_name")
pipeline_id = config.get("pipeline_id")

affected_rows = spark.sql(f"""
    WITH max_timestamp AS (
        SELECT MAX(update_timestamp_utc) AS max_update_timestamp
        FROM {catalog_name}.{schema_name}.train_set
    )
    INSERT INTO {catalog_name}.{schema_name}.train_set
    SELECT *
    FROM {catalog_name}.{schema_name}.extra_train_set
    WHERE update_timestamp_utc > (SELECT max_update_timestamp FROM max_timestamp)
""").first().num_affected_rows

# write into feature table; update online table
if affected_rows > 0:
    spark.sql(f"""
        WITH max_timestamp AS (
            SELECT MAX(update_timestamp_utc) AS max_update_timestamp
            FROM {catalog_name}.{schema_name}.train_set
        )
        INSERT INTO {catalog_name}.{schema_name}.hotel_features
        SELECT Booking_ID, repeated, P_C, P_not_C 
        FROM {catalog_name}.{schema_name}.train_set
        WHERE update_timestamp_utc == (SELECT max_update_timestamp FROM max_timestamp)
""")
    refreshed = 1
    update_response = workspace.pipelines.start_update(
        pipeline_id=pipeline_id, full_refresh=False)
    while True:
        update_info = workspace.pipelines.get_update(pipeline_id=pipeline_id, 
                                update_id=update_response.update_id)
        state = update_info.update.state.value
        if state == 'COMPLETED':
            break
        elif state in ['FAILED', 'CANCELED']:
            raise SystemError("Online table failed to update.")
        elif state == 'WAITING_FOR_RESOURCES':
            print("Pipeline is waiting for resources.")
        else:
            print(f"Pipeline is in {state} state.")
        time.sleep(30)
else:
    refreshed = 0

dbutils.jobs.taskValues.set(key="refreshed", value=refreshed)
