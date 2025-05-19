from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
import time

# Start Spark Session
spark = SparkSession.builder.appName("RandomForestHousingModel").getOrCreate()

# File parameters
file_type = "csv"
file_location = "/user/sruiz85/Group5-Project/county_market_tracker.csv"
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# 2. Load CSV from HDFS
file_location = "/user/sruiz85/Group5-Project/zip_code_market_tracker.csv"
df = spark.read.format("csv") \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .option("sep", ",") \
    .load(file_location)

# 4. Define features and label
feature_cols = [
    'homes_sold', 'pending_sales', 'new_listings', 'inventory',
    'median_list_price', 'median_ppsf', 'median_list_ppsf',
    'avg_sale_to_list', 'sold_above_list', 'median_dom',
    'off_market_in_two_weeks', 'inventory_yoy', 'median_sale_price_yoy'
]
target_col = 'median_sale_price'

# 5. Build model_df
model_df = df.select(feature_cols + [target_col])
model_df = model_df.withColumnRenamed(target_col, "label")

# 6. Fill nulls
model_df = model_df.fillna(0.0, subset=feature_cols + ["label"])

# 7. Split data
train, test = model_df.randomSplit([0.8, 0.2], seed=42)

# 8. Assembler and Scaler
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
scaler = MinMaxScaler(inputCol="features", outputCol="normFeatures")

# Base model
lr_base = RandomForestRegressor(labelCol="label", featuresCol="normFeatures")
pipeline_base = Pipeline(stages=[assembler, scaler, lr_base])
base_model = pipeline_base.fit(train)

# Evaluate base model
base_predictions = base_model.transform(test)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")
rmse_base = evaluator.evaluate(base_predictions, {evaluator.metricName: "rmse"})
r2_base = evaluator.evaluate(base_predictions, {evaluator.metricName: "r2"})

# 9. TrainValidationSplit 1
paramGrid1 = ParamGridBuilder() \
    .addGrid(lr_base.maxDepth, [2, 3]) \
    .addGrid(lr_base.maxBins, [5, 10]) \
    .build()

tvs1 = TrainValidationSplit(
    estimator=pipeline_base,
    estimatorParamMaps=paramGrid1,
    evaluator=evaluator,
    trainRatio=0.8
)
start1 = time.time()
model1 = tvs1.fit(train)
tvs_duration = time.time() - start1

# Evaluate model1
pred1 = model1.transform(test)
rmse_tvs = evaluator.evaluate(pred1, {evaluator.metricName: "rmse"})
r2_tvs = evaluator.evaluate(pred1, {evaluator.metricName: "r2"})
tvs_best_model = model1.bestModel.stages[-1]
tvs_maxDepth = tvs_best_model.getOrDefault("maxDepth")
tvs_maxBins = tvs_best_model.getOrDefault("maxBins")

# 10. TrainValidationSplit 2 (different grid)
paramGrid2 = ParamGridBuilder() \
    .addGrid(lr_base.maxDepth, [3, 5]) \
    .addGrid(lr_base.maxBins, [10, 15]) \
    .build()

tvs2 = TrainValidationSplit(
    estimator=pipeline_base,
    estimatorParamMaps=paramGrid2,
    evaluator=evaluator,
    trainRatio=0.8
)
tvs2.fit(train)  # Fit but not used in comparison for simplicity

# 11. CrossValidator
cv = CrossValidator(
    estimator=pipeline_base,
    estimatorParamMaps=paramGrid1,
    evaluator=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse"),
    numFolds=3
)
start2 = time.time()
model3 = cv.fit(train)
cv_duration = time.time() - start2

# Evaluate model3
pred3 = model3.transform(test)
rmse_cv = evaluator.evaluate(pred3, {evaluator.metricName: "rmse"})
r2_cv = evaluator.evaluate(pred3, {evaluator.metricName: "r2"})
cv_best_model = model3.bestModel.stages[-1]
cv_maxDepth = cv_best_model.getOrDefault("maxDepth")
cv_maxBins = cv_best_model.getOrDefault("maxBins")

# ----------- Final Comparison ------------
base_duration = 0.0  # already trained above without timing

comparison_data = [
    Row(Model="RandomForestRegressor", RMSE=rmse_base, R2=r2_base, Time=base_duration, Best_maxDepth=None, Best_maxBins=None),
    Row(Model="RandomForest (TrainValidationSplit)", RMSE=rmse_tvs, R2=r2_tvs, Time=tvs_duration,
        Best_maxDepth=tvs_maxDepth, Best_maxBins=tvs_maxBins),
    Row(Model="RandomForest (CrossValidator)", RMSE=rmse_cv, R2=r2_cv, Time=cv_duration,
        Best_maxDepth=cv_maxDepth, Best_maxBins=cv_maxBins)
]

comparison_sdf = spark.createDataFrame(comparison_data)
comparison_sdf.show(truncate=False)
