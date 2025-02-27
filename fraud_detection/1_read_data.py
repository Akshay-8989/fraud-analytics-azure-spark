from pyspark.sql import SparkSession
import config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Initialize Spark Session with updated Azure dependencies
    spark = SparkSession.builder \
        .appName("AzureBlobRead") \
        .config("spark.jars", f"{config.SPARK_JARS_PATH}/hadoop-azure-3.3.1.jar,"
                              f"{config.SPARK_JARS_PATH}/azure-storage-blob-12.10.0.jar,"
                              f"{config.SPARK_JARS_PATH}/azure-core-1.15.0.jar") \
        .config("fs.azure.account.key." + config.AZURE_STORAGE_ACCOUNT_NAME + ".blob.core.windows.net",
                config.AZURE_STORAGE_ACCOUNT_KEY) \
        .getOrCreate()

    logger.info("Spark session initialized successfully")

    # Read CSV file from Azure Blob Storage
    df = spark.read.csv(config.AZURE_BLOB_URL, header=True, inferSchema=True)

    logger.info(f"Successfully read data from {config.AZURE_BLOB_URL}")
    logger.info(f"Data schema: {df.schema}")
    logger.info(f"Row count: {df.count()}")

    # Show first few rows
    df.show(5)

    # Save data as a Parquet file for later processing
    df.write.parquet("data/transactions.parquet", mode="overwrite")
    logger.info("Data saved to data/transactions.parquet")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise

finally:
    if 'spark' in locals():
        spark.stop()
        logger.info("Spark session stopped")