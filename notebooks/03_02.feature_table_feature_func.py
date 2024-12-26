# Databricks notebook source
# MAGIC %pip install -r requirements.in

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import yaml
from databricks import feature_engineering
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
with open("../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

catalog_name = config.get("catalog_name")
schema_name = config.get("schema_name")

# COMMAND ----------
feature_table_name = f"{catalog_name}.{schema_name}.hotel_features"
function_name = f"{catalog_name}.{schema_name}.calculate_lead_time"

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

hotel_features_df = train_set[["Booking_ID", "repeated", "P_C", "P_not_C"]]

# COMMAND ----------
# Option 1: feature engineering client
feature_table = fe.create_table(
    name=feature_table_name,
    primary_keys=["Booking_ID"],
    df=train_set[["Booking_ID", "repeated", "P_C", "P_not_C"]],
    description="Hotel booking features",
)

spark.sql(
    f"ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

fe.write_table(
    name=feature_table_name,
    df=test_set[["Booking_ID", "repeated", "P_C", "P_not_C"]],
    mode="merge",
)

# COMMAND ----------
# Option 2: SQL
spark.sql(
    f"""
CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.hotel_features
(Booking_ID STRING NOT NULL, 
repeated INT, 
P_C INT, 
P_not_C INT);"""
)

spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.hotel_features "
    "ADD CONSTRAINT booking_pk PRIMARY KEY(Booking_ID);"
)

spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.hotel_features "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)

spark.sql(
    f"INSERT INTO {catalog_name}.{schema_name}.hotel_features "
    f"SELECT Booking_ID, repeated, P_C, P_not_C FROM {catalog_name}.{schema_name}.train_set"
)

spark.sql(
    f"INSERT INTO {catalog_name}.{schema_name}.hotel_features "
    f"SELECT Booking_ID, repeated, P_C, P_not_C FROM {catalog_name}.{schema_name}.test_set"
)

# COMMAND ----------
spark.sql(
    f"""
CREATE OR REPLACE FUNCTION {function_name}(arrival_date TIMESTAMP, reservation_date TIMESTAMP)
RETURNS INT
LANGUAGE PYTHON AS
$$
return (arrival_date-reservation_date).days
$$"""
)
# COMMAND ----------
