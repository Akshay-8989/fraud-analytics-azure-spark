from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
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

    # Convert categorical columns to numerical
    indexer = StringIndexer(inputCol="transaction_type", outputCol="transaction_type_indexed")
    indexer_model = indexer.fit(df) # Fit and store the model
    df = indexer_model.transform(df)
    logger.info("Converted transaction_type to transaction_type_indexed")

    # Show sample of transformed data
    df.show(5)

    # Save the processed data with the indexed column
    df.write.parquet("data/processed_transactions_with_features.parquet", mode="overwrite")
    logger.info("Processed data saved to data/processed_transactions_with_features.parquet")

    # **SAVE THE INDEXER MODEL**
    indexer_model.write().overwrite().save("models/transaction_type_indexer_model")
    logger.info("Saved StringIndexer model to models/transaction_type_indexer_model")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise

finally:
    if 'spark' in locals():
        spark.stop()
        logger.info("Spark session stopped")
