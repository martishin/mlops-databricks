import mlflow
import argparse
import yaml
from databricks import feature_engineering
from pyspark.sql import SparkSession
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient

from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path",
        action="store",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--git_sha",
        action="store",
        default=None,
        type=str,
        required=True,
    )

    args = parser.parse_args()
    root_path = args.root_path
    git_sha = args.git_sha

    with open(f"/Workspace/{root_path}/files/project_config.yml", "r") as file:
        config = yaml.safe_load(file)

workspace = WorkspaceClient()

fe = feature_engineering.FeatureEngineeringClient()
mlflow.set_registry_uri("databricks-uc")

num_features = config.get("num_features")
cat_features = config.get("cat_features")
target = config.get("target")
parameters = config.get("parameters")
catalog_name = config.get("catalog_name")
schema_name = config.get("schema_name")

spark = SparkSession.builder.getOrCreate()

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop(
    "lead_time", "repeated", "P_C", "P_not_C"
)
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

training_set = fe.create_training_set(
    df=train_set,
    label="booking_status",
    feature_lookups=[
        FeatureLookup(
            table_name=f"{catalog_name}.{schema_name}.hotel_features",
            feature_names=["repeated", "P_C", "P_not_C"],
            lookup_key="Booking_ID",
        ),
        FeatureFunction(
            udf_name=f"{catalog_name}.{schema_name}.calculate_lead_time",
            output_name="lead_time",
            input_bindings={
                "arrival_date": "arrival_date",
                "reservation_date": "reservation_date",
            },
        ),
    ],
    exclude_columns=["update_timestamp_utc", "market_segment_type"]
)

training_df = training_set.load_df().toPandas()

X_train = training_df[num_features + cat_features]
y_train = training_df[target]

X_test = test_set[num_features + cat_features]
y_test = test_set[target]

preprocessor = ColumnTransformer(
    transformers=[("cat", 
                   OneHotEncoder(handle_unknown='ignore'),
                   cat_features)], remainder="passthrough"
)

pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**parameters))]
)

mlflow.set_experiment(experiment_name="/Shared/hotel-cancellations-fe")
with mlflow.start_run(
    tags={"branch": "03_03",
          "git_sha": git_sha},
) as run:
    run_id = run.info.run_id
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)

    print(f"Confusion Matrix: \n{conf}")
    print(f"Classification Report: \n{clf_report}")

    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params({"learning_rate": 0.02, "n_estimators": 500, "num_leaves": 17})
    mlflow.log_metric("accuracy", accuracy)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    model = fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="lightgbm-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )

  
model_version = mlflow.register_model(model_uri=f'runs:/{run_id}/lightgbm-pipeline-model-fe',
                     name=f"{catalog_name}.{schema_name}.hotel_booking_model_fe")

dbutils.jobs.taskValues.set(key="model_version", value=model_version.version)
