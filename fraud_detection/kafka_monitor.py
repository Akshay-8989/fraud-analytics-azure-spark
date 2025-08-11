from kafka import KafkaConsumer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import StringIndexerModel, VectorAssembler
import json
import config  # from your existing project

# ==== CONFIG ====
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "transactions"
MODEL_PATH = "models/fraud_model"
INDEXER_PATH = "models/transaction_type_indexer_model"
FRAUD_THRESHOLD = 0.8
# =================

# Spark session
spark = SparkSession.builder.appName("RealTimeFraudMonitoring").getOrCreate()

# Load models
rf_model = RandomForestClassificationModel.load(MODEL_PATH)
indexer_model = StringIndexerModel.load(INDEXER_PATH)

assembler = VectorAssembler(inputCols=config.FEATURE_COLS, outputCol="features")

# Kafka consumer
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='latest'
)

print("Listening for transactions...")

for msg in consumer:
    data = [msg.value]  # wrap in list for Spark DF
    df = spark.createDataFrame(data)

    # Preprocess (index categorical → vector)
    df = indexer_model.transform(df)
    df = assembler.transform(df)

    # Predict fraud probability
    preds = rf_model.transform(df).withColumn("fraud_probability", col("probability").getItem(1))

    row = preds.collect()[0].asDict()

    if row["fraud_probability"] > FRAUD_THRESHOLD:
        print(f"🚨 FRAUD ALERT: {row}")
    else:
        print(f"✅ Safe transaction: {row}")
