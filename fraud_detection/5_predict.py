from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, StringIndexerModel # Import StringIndexerModel
import logging
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Initialize Spark
    spark = SparkSession.builder.appName("PredictFraud").getOrCreate()
    logger.info("Spark session initialized successfully")

    # Load trained model
    model = RandomForestClassificationModel.load("models/fraud_model")
    logger.info("Model loaded successfully")

    # Load new transaction data (assuming it has a different structure than training data)
    # For demonstration, assuming a new file with raw transactions
    new_data = spark.read.parquet("data/new_transactions.parquet")
    logger.info(f"Loaded new data with {new_data.count()} rows")

    # **LOAD THE INDEXER MODEL**
    indexer_model = StringIndexerModel.load("models/transaction_type_indexer_model")
    logger.info("Loaded StringIndexer model")

    # Apply the same transformations as during training
    # 1. Convert categorical column to index
    new_data = indexer_model.transform(new_data) # Use the loaded model!

    # 2. Create feature vector
    assembler = VectorAssembler(inputCols=config.FEATURE_COLS, outputCol="features")
    new_data = assembler.transform(new_data)
    logger.info("Feature preprocessing completed")

    # Make predictions
    predictions = model.transform(new_data)

    # Calculate prediction probability
    predictions = predictions.withColumn(
        "fraud_probability",
        predictions.probability.getItem(1)
    )

    # Show predictions
    predictions.select(
        "transaction_id",
        "prediction",
        "fraud_probability"
    ).show(10)

    logger.info("Predictions generated successfully")

    # Save predictions
    predictions.select(
        "transaction_id",
        "prediction",
        "fraud_probability"
    ).write.parquet("data/fraud_predictions.parquet", mode="overwrite")

    logger.info("Predictions saved to data/fraud_predictions.parquet")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise

finally:
    if 'spark' in locals():
        spark.stop()
        logger.info("Spark session stopped")
