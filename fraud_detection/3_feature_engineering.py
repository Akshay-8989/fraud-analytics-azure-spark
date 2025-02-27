from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import logging
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Initialize Spark
    spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()
    logger.info("Spark session initialized successfully")

    # Load processed data with indexed features
    df = spark.read.parquet("data/processed_transactions_with_features.parquet")
    logger.info(f"Loaded data with {df.count()} rows")

    # Define features using the config
    assembler = VectorAssembler(inputCols=config.FEATURE_COLS, outputCol="features")
    logger.info(f"Creating feature vector from columns: {config.FEATURE_COLS}")

    # Transform dataset
    df = assembler.transform(df)
    logger.info("Feature vector created successfully")

    # Save for training
    df.select("transaction_id", "features", "fraud_label").write.parquet("data/ml_ready_transactions.parquet", mode="overwrite")
    logger.info("ML-ready data saved to data/ml_ready_transactions.parquet")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise

finally:
    if 'spark' in locals():
        spark.stop()
        logger.info("Spark session stopped")