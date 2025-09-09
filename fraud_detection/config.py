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

# config.py

FEATURE_COLS = [
    "transaction_type_indexed",
    "amount",
    "hour_of_day",
    "day_of_week",
    "geo_location_mismatch",
    "device_id_indexed",
    "balance_diff_orig",
    "balance_diff_dest",
    "past_transactions_count",
    "is_large_transaction"
]

LABEL_COL = "isFraud"
