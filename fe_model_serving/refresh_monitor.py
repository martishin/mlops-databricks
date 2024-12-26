import argparse

import yaml
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

if __name__ == "__main__":
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

catalog_name = config.get("catalog_name")
schema_name = config.get("schema_name")

spark = SparkSession.builder.getOrCreate()

workspace = WorkspaceClient()

inf_table = spark.sql(
    f"SELECT * FROM {catalog_name}.{schema_name}.model_serving_payload_payload "
    f"WHERE timestamp_ms > (SELECT MAX(timestamp_ms) FROM {catalog_name}.{schema_name}.model_monitoring)"
)

request_schema = StructType(
    [
        StructField(
            "dataframe_records",
            ArrayType(
                StructType(
                    [
                        StructField("Booking_ID", StringType(), True),
                        StructField("number_of_adults", IntegerType(), True),
                        StructField("number_of_children", IntegerType(), True),
                        StructField("number_of_weekend_nights", IntegerType(), True),
                        StructField("number_of_week_nights", IntegerType(), True),
                        StructField("car_parking_space", IntegerType(), True),
                        StructField("average_price", DoubleType(), True),
                        StructField("special_requests", IntegerType(), True),
                        StructField("arrival_date", StringType(), True),
                        StructField("reservation_date", StringType(), True),
                        StructField("type_of_meal", StringType(), True),
                        StructField("room_type", StringType(), True),
                    ]
                )
            ),
            True,
        )
    ]
)

response_schema = StructType(
    [
        StructField("predictions", ArrayType(IntegerType()), True),
        StructField(
            "databricks_output",
            StructType(
                [
                    StructField("trace", StringType(), True),
                    StructField("databricks_request_id", StringType(), True),
                ]
            ),
            True,
        ),
    ]
)


inf_table_parsed = inf_table.withColumn(
    "parsed_request", F.from_json(F.col("request"), request_schema)
)

inf_table_parsed = inf_table_parsed.withColumn(
    "parsed_response", F.from_json(F.col("response"), response_schema)
)

df_exploded = inf_table_parsed.withColumn(
    "record", F.explode(F.col("parsed_request.dataframe_records"))
)

df_final = df_exploded.select(
    F.from_unixtime(F.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
    "timestamp_ms",
    "databricks_request_id",
    "execution_time_ms",
    F.col("record.Booking_ID").alias("Booking_ID"),
    F.col("record.number_of_adults").alias("number_of_adults"),
    F.col("record.number_of_children").alias("number_of_children"),
    F.col("record.number_of_weekend_nights").alias("number_of_weekend_nights"),
    F.col("record.number_of_week_nights").alias("number_of_week_nights"),
    F.col("record.car_parking_space").alias("car_parking_space"),
    F.col("record.average_price").alias("average_price"),
    F.col("record.special_requests").alias("special_requests"),
    F.col("record.arrival_date").alias("arrival_date"),
    F.col("record.reservation_date").alias("reservation_date"),
    F.col("record.type_of_meal").alias("type_of_meal"),
    F.col("record.room_type").alias("room_type"),
    F.col("parsed_response.predictions")[0].alias("prediction"),
    F.lit("hotel-cancellations-model-fe").alias("model_name"),
)

test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")
extra_set = spark.table(f"{catalog_name}.{schema_name}.extra_set")

df_final_with_status = (
    df_final.join(
        test_set.select("Booking_ID", "booking_status"), on="Booking_ID", how="left"
    )
    .withColumnRenamed("booking_status", "booking_status_test_set")
    .join(extra_set.select("Booking_ID", "booking_status"), on="Booking_ID", how="left")
    .withColumnRenamed("booking_status", "booking_status_extra_set")
    .select(
        "*",
        F.coalesce(
            F.col("booking_status_test_set"), F.col("booking_status_extra_set")
        ).alias("booking_status"),
    )
    .drop("booking_status_test_set", "booking_status_extra_set")
    .withColumn("booking_status", F.col("booking_status").cast("int"))
)

df_final_with_status.write.format("delta").mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.model_monitoring"
)

workspace.quality_monitors.run_refresh(
    table_name=f"{catalog_name}.{schema_name}.model_monitoring"
)
