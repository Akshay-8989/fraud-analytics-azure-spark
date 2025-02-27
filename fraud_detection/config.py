import os

AZURE_STORAGE_ACCOUNT_NAME = "your-storage-account-name"
# Use environment variable for the key (more secure)
AZURE_STORAGE_ACCOUNT_KEY = os.environ.get("AZURE_STORAGE_KEY", "your_account_key")
CONTAINER_NAME = "your-container-name"
BLOB_FILE_NAME = "file-name.csv"

AZURE_BLOB_URL = f"wasbs://{CONTAINER_NAME}@{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{BLOB_FILE_NAME}"

# Update this to your specific Spark installation path
SPARK_JARS_PATH = "jars-folder-path"   # example - "C:/spark/spark-3.5.4-bin-hadoop3/jars"

# Define common feature columns for reuse
FEATURE_COLS = ["amount", "transaction_type_indexed", "account_age", "num_prev_transactions"]