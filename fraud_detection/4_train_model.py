from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Initialize Spark
    spark = SparkSession.builder.appName("TrainModel").getOrCreate()
    logger.info("Spark session initialized successfully")

    # Load processed data
    df = spark.read.parquet("data/ml_ready_transactions.parquet")
    logger.info(f"Loaded data with {df.count()} rows")

    # Split data into training and test sets
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    logger.info(f"Training set: {train_df.count()} rows, Test set: {test_df.count()} rows")

    # Initialize RandomForest model
    rf = RandomForestClassifier(featuresCol="features", labelCol="fraud_label", numTrees=50)

    # Create parameter grid for hyperparameter tuning
    param_grid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [20, 50, 100]) \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
        .build()
    logger.info("Created parameter grid for cross-validation")

    # Define evaluator
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="fraud_label", 
        metricName="areaUnderROC"
    )

    # Perform cross-validation
    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=param_grid,
        evaluator=auc_evaluator,
        numFolds=3
    )
    
    logger.info("Starting cross-validation for model training...")
    cv_model = cv.fit(train_df)
    logger.info("Cross-validation complete")

    # Evaluate on test set
    predictions = cv_model.bestModel.transform(test_df)
    
    # Multiple evaluation metrics
    auc = auc_evaluator.evaluate(predictions)
    
    # Add more evaluation metrics
    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="fraud_label", 
        predictionCol="prediction", 
        metricName="weightedPrecision"
    )
    precision = precision_evaluator.evaluate(predictions)
    
    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="fraud_label", 
        predictionCol="prediction", 
        metricName="weightedRecall"
    )
    recall = recall_evaluator.evaluate(predictions)
    
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="fraud_label", 
        predictionCol="prediction", 
        metricName="f1"
    )
    f1 = f1_evaluator.evaluate(predictions)
    
    logger.info(f"Test AUC: {auc:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")

    # Get feature importance
    best_model = cv_model.bestModel
    feature_importances = best_model.featureImportances
    logger.info(f"Feature importances: {feature_importances}")

    # Save the trained model
    best_model.write().overwrite().save("models/fraud_model")
    logger.info("Model saved to models/fraud_model")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise

finally:
    if 'spark' in locals():
        spark.stop()
        logger.info("Spark session stopped")