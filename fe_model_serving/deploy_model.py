import yaml
import argparse
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput

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

version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

workspace = WorkspaceClient()

catalog_name = config.get("catalog_name")
schema_name = config.get("schema_name")

workspace.serving_endpoints.update_config_and_wait(
    name="hotel-cancellations-model-fe",
    served_entities=[
        ServedEntityInput(
            entity_name=f"{catalog_name}.{schema_name}.hotel_booking_model_fe",
            scale_to_zero_enabled=True,
            workload_size="Small",
            entity_version=version,
        )
    ],
)
