from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col, hour, dayofweek, when
import logging
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Initialize Spark
    spark = SparkSession.builder.appName("ProcessData").getOrCreate()
    logger.info("Spark session initialized successfully")

    # Load data
    df = spark.read.parquet("data/transactions.parquet")
    logger.info(f"Loaded data with {df.count()} rows")

    # Show schema
    df.printSchema()

    # === Feature Engineering ===

    # 1. Transaction type → numerical
    type_indexer = StringIndexer(inputCol="transaction_type", outputCol="transaction_type_indexed")
    type_indexer_model = type_indexer.fit(df)
    df = type_indexer_model.transform(df)
    logger.info("Converted transaction_type to transaction_type_indexed")

    # 2. Device ID → numerical
    device_indexer = StringIndexer(inputCol="device_id", outputCol="device_id_indexed")
    device_indexer_model = device_indexer.fit(df)
    df = device_indexer_model.transform(df)
    logger.info("Converted device_id to device_id_indexed")

    # 3. Time features
    df = df.withColumn("hour_of_day", hour(col("transaction_time")))
    df = df.withColumn("day_of_week", dayofweek(col("transaction_time")))

    # 4. Geo-location mismatch (1 if origin != destination)
    df = df.withColumn(
        "geo_location_mismatch",
        when(col("origin_location") != col("destination_location"), 1).otherwise(0)
    )

    # 5. Balance differences
    df = df.withColumn("balance_diff_orig", col("oldbalanceOrg") - col("newbalanceOrig"))
    df = df.withColumn("balance_diff_dest", col("newbalanceDest") - col("oldbalanceDest"))

    # 6. Flag large transactions (e.g., > 10,000)
    df = df.withColumn("is_large_transaction", when(col("amount") > 10000, 1).otherwise(0))

    # (Optional) 7. Past transactions count (dummy column for now – real version needs history aggregation)
    if "past_transactions_count" not in df.columns:
        df = df.withColumn("past_transactions_count", when(col("transaction_id").isNotNull(), 1).otherwise(0))

    logger.info("Feature engineering completed")

    # Show sample
    df.select(
        "transaction_id",
        "transaction_type_indexed",
        "device_id_indexed",
        "hour_of_day",
        "day_of_week",
        "geo_location_mismatch",
        "balance_diff_orig",
        "balance_diff_dest",
        "is_large_transaction",
        "past_transactions_count"
    ).show(5)

    # Save processed data
    df.write.parquet("data/processed_transactions_with_features.parquet", mode="overwrite")
    logger.info("Processed data saved to data/processed_transactions_with_features.parquet")

    # Save indexer models
    type_indexer_model.write().overwrite().save("models/transaction_type_indexer_model")
    device_indexer_model.write().overwrite().save("models/device_id_indexer_model")
    logger.info("Saved indexer models for transaction_type and device_id")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise

finally:
    if 'spark' in locals():
        spark.stop()
        logger.info("Spark session stopped")
