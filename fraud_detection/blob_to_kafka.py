from azure.storage.blob import BlobServiceClient
from kafka import KafkaProducer
import pandas as pd
import json
import time

# ==== CONFIG ====
BLOB_CONNECTION_STRING = "<your-azure-blob-connection-string>"
BLOB_CONTAINER_NAME = "<container-name>"
BLOB_FILE_NAME = "<transactions.csv>"
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "transactions"
SEND_INTERVAL_SEC = 2  # delay between sending rows (simulate streaming)
# =================

# Azure Blob client
blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=BLOB_FILE_NAME)

# Download CSV from blob
with open("temp_transactions.csv", "wb") as file:
    file.write(blob_client.download_blob().readall())

print(f"Downloaded {BLOB_FILE_NAME} from Azure Blob.")

# Read into DataFrame
df = pd.read_csv("temp_transactions.csv")

# Kafka producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Send each transaction to Kafka
for _, row in df.iterrows():
    producer.send(KAFKA_TOPIC, row.to_dict())
    print(f"Sent: {row.to_dict()}")
    time.sleep(SEND_INTERVAL_SEC)

producer.flush()
print("All transactions sent.")
