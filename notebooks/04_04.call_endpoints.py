# Databricks notebook source
import time

import requests

# COMMAND ----------

token = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)
host = spark.conf.get("spark.databricks.workspaceUrl")


# COMMAND ----------

start_time = time.time()
serving_endpoint = (
    f"https://{host}/serving-endpoints/hotel-cancellations-preds/invocations"
)
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": [{"Booking_ID": "INN00005"}]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

start_time = time.time()
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_split": {"columns": ["Booking_ID"], "data": [["INN00005"]]}},
)
end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

start_time = time.time()
dataframe_records = [
    {
        "number_of_adults": 1,
        "number_of_children": 0,
        "number_of_weekend_nights": 1,
        "number_of_week_nights": 2,
        "car_parking_space": 1,
        "average_price": 120.5,
        "special_requests": 0,
        "lead_time": 20,
        "repeated": 0,
        "P_C": 0,
        "P_not_C": 1,
        "type_of_meal": "Meal Plan 1",
        "room_type": "Room_Type 1",
    }
]
model_serving_endpoint = (
    f"https://{host}/serving-endpoints/hotel-cancellations-model/invocations"
)
response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------
start_time = time.time()
dataframe_records = [
    {
        "Booking_ID": "INN00005",
        "number_of_adults": 1,
        "number_of_children": 0,
        "number_of_weekend_nights": 1,
        "number_of_week_nights": 2,
        "car_parking_space": 1,
        "average_price": 120.5,
        "special_requests": 0,
        "arrival_date": "2024-09-18",
        "reservation_date": "2024-09-08",
        "type_of_meal": "Meal Plan 1",
        "room_type": "Room_Type 1",
    }
]
fe_model_serving_endpoint = (
    f"https://{host}/serving-endpoints/hotel-cancellations-model-fe/invocations"
)
response = requests.post(
    f"{fe_model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")
