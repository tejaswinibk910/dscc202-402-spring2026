# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer: ML Inference for Sentiment Prediction
# MAGIC
# MAGIC ## Purpose
# MAGIC Apply pre-trained sentiment model to predict tweet sentiment.
# MAGIC Enrich data with ML predictions for comparison with ground truth labels.
# MAGIC
# MAGIC ## Requirements
# MAGIC - Load model from Unity Catalog: workspace.default.tweet_sentiment_model
# MAGIC - Create Spark UDF for distributed ML inference
# MAGIC - Map model labels (LABEL_0/1/2) to sentiment strings (negative/neutral/positive)
# MAGIC - Scale confidence scores to 0-100 range
# MAGIC - Create binary sentiment indicators for classification metrics
# MAGIC
# MAGIC ## Expected Output
# MAGIC Delta table: `tweets_gold`
# MAGIC - Row count matches silver
# MAGIC - predicted_score in range 0-100
# MAGIC - predicted_sentiment: negative/neutral/positive
# MAGIC - Binary IDs (0 or 1) for ground truth and predictions
# MAGIC
# MAGIC ## Model Information
# MAGIC - Model: twitter-roberta-base-sentiment
# MAGIC - Output: Struct with label (string) and score (double)
# MAGIC - Labels: LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive
# MAGIC
# MAGIC ## Reference
# MAGIC See Lab 0.5 (MLops) for MLflow model loading and Spark UDF patterns

# COMMAND ----------

# TODO: Import necessary libraries
# You will need:
# - pyspark.pipelines (as dp)
# - pyspark.sql.types and pyspark.sql.functions
# - mlflow for model loading
import pyspark.pipelines as dp
import mlflow
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql.functions import col, when

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Create Gold Streaming Table
# MAGIC
# MAGIC TODO: Define streaming table "tweets_gold" with descriptive comment

# COMMAND ----------

# TODO: Create streaming table definition
dp.create_streaming_table(
    "tweets_gold",
    comment="Tweet sentiment predictions from ML model with ground truth labels"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Configure MLflow Registry
# MAGIC
# MAGIC TODO: Set MLflow registry to Unity Catalog
# MAGIC Use: mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# TODO: Configure MLflow registry
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Define Model Output Schema
# MAGIC
# MAGIC TODO: Define StructType for model output with fields:
# MAGIC - label (StringType): LABEL_0, LABEL_1, or LABEL_2
# MAGIC - score (DoubleType): Confidence score 0.0-1.0

# COMMAND ----------

# TODO: Define model output schema
model_output_schema = StructType([
    StructField("label", StringType(), True),
    StructField("score", DoubleType(), True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Load Model and Create Spark UDF
# MAGIC
# MAGIC TODO: Load sentiment model from Unity Catalog and create Spark UDF
# MAGIC - Model URI: "models:/workspace.default.tweet_sentiment_model/1"
# MAGIC - Use: mlflow.pyfunc.spark_udf(spark, model_uri, result_type)
# MAGIC
# MAGIC This enables distributed ML inference across all Spark executors.

# COMMAND ----------

sentiment_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri="runs:/33b93de27ef447ac84a04e94a438d7df/model",
    result_type=model_output_schema,
    env_manager="local",
    params={"batch_size": 32}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5: Define Gold Transformation Flow
# MAGIC
# MAGIC TODO: Create @dp.append_flow function that:
# MAGIC 1. Reads from tweets_silver streaming table
# MAGIC 2. Applies model UDF to cleaned_text column
# MAGIC 3. Extracts label from model output struct
# MAGIC 4. Extracts score and scales to 0-100 (multiply by 100)
# MAGIC 5. Maps labels to sentiment strings:
# MAGIC    - LABEL_0 → "negative"
# MAGIC    - LABEL_1 → "neutral"
# MAGIC    - LABEL_2 → "positive"
# MAGIC 6. Creates binary sentiment_id (0=negative, 1=positive/neutral)
# MAGIC 7. Creates binary predicted_sentiment_id (0=negative, 1=positive/neutral)
# MAGIC 8. Selects final columns (9 total)
# MAGIC
# MAGIC Reference: Lab 0.5 for model UDF application and struct parsing

# COMMAND ----------

@dp.append_flow(target="tweets_gold")
def transform_gold():
    return (
        spark.readStream
            .option("maxBytesPerTrigger", "10mb")
            .table("tweets_silver")
            .repartition(8)
            .withColumn("model_output", sentiment_udf(col("cleaned_text")))
            .withColumn("predicted_label", col("model_output.label"))
            .withColumn("predicted_score", col("model_output.score") * 100)
            .withColumn("predicted_sentiment",
                when(col("predicted_label") == "NEGATIVE", "negative")
                .otherwise("positive")
            )
            .withColumn("sentiment_id",
                when(col("sentiment") == "0", 0)
                .when(col("sentiment") == "4", 1)
                .otherwise(None).cast(IntegerType())
            )
            .withColumn("predicted_sentiment_id",
                when(col("predicted_sentiment") == "negative", 0).otherwise(1).cast(IntegerType())
            )
            .select(
                "timestamp", "mention", "cleaned_text", "text", "sentiment",
                "predicted_sentiment", "predicted_score",
                "sentiment_id", "predicted_sentiment_id"
            )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation
# MAGIC
# MAGIC After pipeline execution, verify:
# MAGIC - Row count matches silver
# MAGIC - predicted_score: 0-100 range
# MAGIC - predicted_sentiment: "negative", "neutral", or "positive"
# MAGIC - sentiment_id and predicted_sentiment_id: 0 or 1
# MAGIC - All rows have predictions (no nulls)
