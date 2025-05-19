# 1. Imports
from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import time

# 2. Start Spark Session
spark = SparkSession.builder.appName("OptimizedLinearRegression").getOrCreate()

# 3. Load Dataset
file_location = "/user/sruiz85/Group5-Project/state_market_tracker.csv"
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("sep", ",") \
    .load(file_location)

# 4. Feature and Label Columns
feature_cols = [
    'homes_sold', 'pending_sales', 'new_listings', 'inventory',
    'median_list_price', 'median_ppsf', 'median_list_ppsf',
    'avg_sale_to_list', 'sold_above_list', 'median_dom',
    'off_market_in_two_weeks', 'inventory_yoy', 'median_sale_price_yoy'
]
target_col = 'median_sale_price'
df = df.select(feature_cols + [target_col]).withColumnRenamed(target_col, "label")

# 5. Assemble and Scale Features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
scaler = MinMaxScaler(inputCol="features", outputCol="normFeatures")

# 6. Split Data (More for training)
train_df, test_df = df.randomSplit([0.85, 0.15], seed=42)

# 7. Base Model (No Tuning)
base_pipeline = Pipeline(stages=[assembler, scaler, LinearRegression(featuresCol="normFeatures", labelCol="label")])
start_base = time.time()
base_model = base_pipeline.fit(train_df)
base_duration = time.time() - start_base
base_pred = base_model.transform(test_df)
rmse_base = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse").evaluate(base_pred)
r2_base = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(base_pred)

# 8. Extended Param Grid
lr = LinearRegression(featuresCol="normFeatures", labelCol="label")
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.001, 0.01, 0.1, 0.3, 0.5]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0]) \
    .build()

pipeline = Pipeline(stages=[assembler, scaler, lr])

# 9. TrainValidationSplit
tvs = TrainValidationSplit(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse"),
    trainRatio=0.8
)
start_tvs = time.time()
tvs_model = tvs.fit(train_df)
tvs_duration = time.time() - start_tvs
tvs_pred = tvs_model.transform(test_df)
rmse_tvs = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse").evaluate(tvs_pred)
r2_tvs = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(tvs_pred)
tvs_best = tvs_model.bestModel.stages[-1]

# 10. CrossValidator (5-fold CV)
cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse"),
    numFolds=5
)
start_cv = time.time()
cv_model = cv.fit(train_df)
cv_duration = time.time() - start_cv
cv_pred = cv_model.transform(test_df)
rmse_cv = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse").evaluate(cv_pred)
r2_cv = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(cv_pred)
cv_best = cv_model.bestModel.stages[-1]

# 11. Summary Table
summary_data = [
    Row(Model="LinearRegression (Base)", RMSE=rmse_base, R2=r2_base, Time=base_duration,
        Best_regParam=None, Best_elasticNet=None),
    Row(Model="LinearRegression (TrainValidationSplit)", RMSE=rmse_tvs, R2=r2_tvs, Time=tvs_duration,
        Best_regParam=tvs_best._java_obj.getRegParam(),
        Best_elasticNet=tvs_best._java_obj.getElasticNetParam()),
    Row(Model="LinearRegression (CrossValidator)", RMSE=rmse_cv, R2=r2_cv, Time=cv_duration,
        Best_regParam=cv_best._java_obj.getRegParam(),
        Best_elasticNet=cv_best._java_obj.getElasticNetParam())
]

summary_df = spark.createDataFrame(summary_data)
print("\n Optimized Linear Regression Model Comparison:")
summary_df.show(truncate=False)
