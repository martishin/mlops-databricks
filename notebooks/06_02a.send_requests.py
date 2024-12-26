# Databricks notebook source
import time

import yaml
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient

workspace = WorkspaceClient()

with open("../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

catalog_name = config.get("catalog_name")
schema_name = config.get("schema_name")
pipeline_id = config.get("pipeline_id")

spark = DatabricksSession.builder.getOrCreate()

test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
extra_set = spark.table(f"{catalog_name}.{schema_name}.extra_set").toPandas()
df_test = test_set

df_extra = extra_set[extra_set["market_segment_type"] == "Offline"]

# COMMAND ----------
spark.sql(
    f"""
        INSERT INTO {catalog_name}.{schema_name}.hotel_features
        SELECT Booking_ID, repeated, P_C, P_not_C FROM 
        {catalog_name}.{schema_name}.extra_set WHERE market_segment_type=='Offline'
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
        print("Pipeline is waiting for resources...")
    else:
        print(f"Pipeline is in {state} state...")
    time.sleep(30)


# COMMAND ----------
def send_request(row):
    workspace.serving_endpoints.query(
        name="hotel-cancellations-model-fe",
        dataframe_records=[
            {
                "Booking_ID": row["Booking_ID"],
                "number_of_adults": row["number_of_adults"],
                "number_of_children": row["number_of_children"],
                "number_of_weekend_nights": row["number_of_weekend_nights"],
                "number_of_week_nights": row["number_of_week_nights"],
                "car_parking_space": row["car_parking_space"],
                "average_price": row["average_price"],
                "special_requests": row["special_requests"],
                "arrival_date": row["arrival_date"].strftime("%Y-%m-%d %H:%M:%S"),
                "reservation_date": row["reservation_date"].strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "type_of_meal": row["type_of_meal"],
                "room_type": row["room_type"],
            }
        ],
    )


# COMMAND ----------
# Loop over rows and send requests
for index, row in df_test.iterrows():
    print(f"sending test df request for index {index}")
    send_request(row)
    time.sleep(0.2)

# COMMAND ----------
for index, row in df_extra.iterrows():
    print(f"sending extra df request for index {index}")
    send_request(row)
    time.sleep(0.2)
# COMMAND ----------
