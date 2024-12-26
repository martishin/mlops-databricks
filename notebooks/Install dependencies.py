# Databricks notebook source
# Step 1: Install uv using pip
%pip install uv

# Step 2: Configure UV to use Databricks' virtual environment
import os
os.environ["UV_PROJECT_ENVIRONMENT"] = os.environ["VIRTUAL_ENV"]

# Step 3: Use shell commands to run uv and install dependencies from pyproject.toml
!uv pip install -r ../pyproject.toml --all-extras

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

