from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StringIndexer, OneHotEncoder
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import log1p, month, year
from pyspark.sql import Row
import time

# Start Spark session
spark = SparkSession.builder.appName("RF_County_Log_vs_Raw").getOrCreate()

# Load dataset
df = spark.read.csv("/user/sruiz85/Group5-Project/county_market_tracker.csv", header=True, inferSchema=True)
df = df.na.drop(subset=["median_list_price"])

# Extract date features
df = df.withColumn("month", month("period_begin"))
df = df.withColumn("year", year("period_begin"))

# Encode region
region_indexer = StringIndexer(inputCol="region", outputCol="region_index", handleInvalid="keep")
region_encoder = OneHotEncoder(inputCol="region_index", outputCol="region_vec")

# Feature columns
base_features = [
    "homes_sold", "pending_sales", "new_listings", "inventory",
    "median_ppsf", "median_list_ppsf", "avg_sale_to_list",
    "sold_above_list", "median_dom", "off_market_in_two_weeks",
    "inventory_yoy", "median_sale_price_yoy", "new_listings_yoy",
    "pending_sales_yoy", "months_of_supply", "price_drops",
    "month", "year"
]
all_features = base_features + ["region_vec"]

# Shared stages
assembler = VectorAssembler(inputCols=all_features, outputCol="features", handleInvalid="skip")
scaler = MinMaxScaler(inputCol="features", outputCol="normFeatures")
rf = RandomForestRegressor(labelCol="label", featuresCol="normFeatures")

evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

paramGrid = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [4, 6, 8]) \
    .addGrid(rf.numTrees, [30, 50, 70]) \
    .build()

# Function to run pipeline and return results
def run_rf_pipeline(df_input, label_name, tag):
    print(f"\n=== Running Random Forest ({tag}) ===")
    df = df_input.withColumnRenamed(label_name, "label")
    df = df.fillna(0.0, subset=base_features)
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    
    pipeline = Pipeline(stages=[region_indexer, region_encoder, assembler, scaler, rf])
    results = []

    # Baseline
    start_train = time.time()
    baseline_model = pipeline.fit(train)
    train_time = time.time() - start_train
    start_pred = time.time()
    baseline_pred = baseline_model.transform(test)
    pred_time = time.time() - start_pred
    rmse = evaluator_rmse.evaluate(baseline_pred)
    r2 = evaluator_r2.evaluate(baseline_pred)
    results.append(Row(Model=f"{tag} (Baseline)", RMSE=rmse, R2=r2, TrainTime=train_time, PredictTime=pred_time))

    # TSV
    tvs = TrainValidationSplit(estimator=pipeline, estimatorParamMaps=paramGrid,
                               evaluator=evaluator_rmse, trainRatio=0.8)
    start_train = time.time()
    tvs_model = tvs.fit(train)
    train_time = time.time() - start_train
    start_pred = time.time()
    tvs_pred = tvs_model.transform(test)
    pred_time = time.time() - start_pred
    rmse = evaluator_rmse.evaluate(tvs_pred)
    r2 = evaluator_r2.evaluate(tvs_pred)
    results.append(Row(Model=f"{tag} (TVS)", RMSE=rmse, R2=r2, TrainTime=train_time, PredictTime=pred_time))

    # CV
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid,
                        evaluator=evaluator_rmse, numFolds=3)
    start_train = time.time()
    cv_model = cv.fit(train)
    train_time = time.time() - start_train
    start_pred = time.time()
    cv_pred = cv_model.transform(test)
    pred_time = time.time() - start_pred
    rmse = evaluator_rmse.evaluate(cv_pred)
    r2 = evaluator_r2.evaluate(cv_pred)
    results.append(Row(Model=f"{tag} (CV)", RMSE=rmse, R2=r2, TrainTime=train_time, PredictTime=pred_time))

    # Save predictions and return results
    cv_pred.select("label", "prediction").write.csv(
        f"/user/sruiz85/Group5-Project/county_rf_{tag.lower().replace(' ', '_')}_predictions", 
        header=True, mode="overwrite"
    )
    return results

# === Run for raw target
raw_results = run_rf_pipeline(df.withColumn("label", df["median_list_price"]), "label", "Raw Target")

# === Run for log-transformed target
log_df = df.withColumn("label", log1p(df["median_list_price"]))
log_results = run_rf_pipeline(log_df, "label", "Log Target")

# === Combine and write summary
all_results = spark.createDataFrame(raw_results + log_results)
print("\n=== RANDOM FOREST COMPARISON: LOG vs RAW ===")
all_results.show(truncate=False)
all_results.write.csv("/user/sruiz85/Group5-Project/county_rf_log_vs_raw_summary", header=True, mode="overwrite")

spark.stop()
