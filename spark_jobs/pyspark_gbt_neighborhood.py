# 1. Import libraries
from pyspark.sql import SparkSession, Row
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
import time

# Start Spark Session
spark = SparkSession.builder.appName("GradientBoostingModel").getOrCreate()

# 2. Load CSV from HDFS
file_location = "/user/sruiz85/Group5-Project/neighborhood_market_tracker.csv"
df = spark.read.format("csv") \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .option("sep", ",") \
    .load(file_location)

# 3. Feature and label columns
feature_cols = [
    'homes_sold', 'pending_sales', 'new_listings', 'inventory',
    'median_list_price', 'median_ppsf', 'median_list_ppsf',
    'avg_sale_to_list', 'sold_above_list', 'median_dom',
    'off_market_in_two_weeks', 'inventory_yoy', 'median_sale_price_yoy'
]
target_col = 'median_sale_price'

# 4. Prepare dataset
model_df = df.select(feature_cols + [target_col])
model_df = model_df.withColumnRenamed(target_col, "label")

# 5. Split into train/test sets
train, test = model_df.randomSplit([0.8, 0.2], seed=42)

# 6. Assemble features for base model
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
train_assembled = assembler.transform(train)
test_assembled = assembler.transform(test)

# 7. Train base GBT model
gb = GBTRegressor(featuresCol="features", labelCol="label")
start = time.time()
gb_model = gb.fit(train_assembled)
base_duration = time.time() - start

# 8. Evaluate base model
pred_base = gb_model.transform(test_assembled)
evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
rmse = evaluator_rmse.evaluate(pred_base)
r2 = evaluator_r2.evaluate(pred_base)

print(f"Base Model RMSE: {rmse:.2f}, R²: {r2:.4f}")
print("\nFeature Importances:")
for feature, importance in zip(feature_cols, gb_model.featureImportances.toArray()):
    print(f"{feature}: {importance:.4f}")

# 9. Define full pipeline components
scaler = MinMaxScaler(inputCol="features", outputCol="normFeatures")
gbt = GBTRegressor(labelCol="label", featuresCol="normFeatures", seed=42)

# 10. TrainValidationSplit 1
paramGrid1 = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [3, 5]) \
    .addGrid(gbt.maxIter, [5, 10]) \
    .build()

pipeline1 = Pipeline(stages=[assembler, scaler, gbt])
start = time.time()
tvs1 = TrainValidationSplit(estimator=pipeline1, estimatorParamMaps=paramGrid1,
                            evaluator=evaluator_rmse, trainRatio=0.8)
model1 = tvs1.fit(train)
tvs_duration = time.time() - start

best_model_tvs = model1.bestModel.stages[-1]
tvs_maxDepth = best_model_tvs.getOrDefault("maxDepth")
tvs_maxIter = best_model_tvs.getOrDefault("maxIter")
pred_tvs = model1.transform(test)
rmse_tvs = evaluator_rmse.evaluate(pred_tvs)
r2_tvs = evaluator_r2.evaluate(pred_tvs)

# 11. TrainValidationSplit 2 (larger grid)
paramGrid2 = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [5, 7]) \
    .addGrid(gbt.maxIter, [10, 20]) \
    .build()

pipeline2 = Pipeline(stages=[assembler, scaler, gbt])
start = time.time()
tvs2 = TrainValidationSplit(estimator=pipeline2, estimatorParamMaps=paramGrid2,
                            evaluator=evaluator_rmse, trainRatio=0.8)
model2 = tvs2.fit(train)
tvs2_duration = time.time() - start

best_model_tvs2 = model2.bestModel.stages[-1]
tvs2_maxDepth = best_model_tvs2.getOrDefault("maxDepth")
tvs2_maxIter = best_model_tvs2.getOrDefault("maxIter")
pred_tvs2 = model2.transform(test)
rmse_tvs2 = evaluator_rmse.evaluate(pred_tvs2)
r2_tvs2 = evaluator_r2.evaluate(pred_tvs2)

# 12. CrossValidator using paramGrid1
pipeline3 = Pipeline(stages=[assembler, scaler, gbt])
cv = CrossValidator(estimator=pipeline3, estimatorParamMaps=paramGrid1,
                    evaluator=evaluator_rmse, numFolds=3)
start = time.time()
model3 = cv.fit(train)
cv_duration = time.time() - start

best_model_cv = model3.bestModel.stages[-1]
cv_maxDepth = best_model_cv.getOrDefault("maxDepth")
cv_maxIter = best_model_cv.getOrDefault("maxIter")
pred_cv = model3.transform(test)
rmse_cv = evaluator_rmse.evaluate(pred_cv)
r2_cv = evaluator_r2.evaluate(pred_cv)

# 13. Print final comparisons
comparison_data = [
    Row(Model="GBTRegressor (Base)", RMSE=rmse, R2=r2, Time=base_duration, Best_maxDepth=None, Best_maxIter=None),
    Row(Model="TrainValidationSplit (Simple Grid)", RMSE=rmse_tvs, R2=r2_tvs, Time=tvs_duration,
        Best_maxDepth=tvs_maxDepth, Best_maxIter=tvs_maxIter),
    Row(Model="TrainValidationSplit (Extended Grid)", RMSE=rmse_tvs2, R2=r2_tvs2, Time=tvs2_duration,
        Best_maxDepth=tvs2_maxDepth, Best_maxIter=tvs2_maxIter),
    Row(Model="CrossValidator (Grid 1)", RMSE=rmse_cv, R2=r2_cv, Time=cv_duration,
        Best_maxDepth=cv_maxDepth, Best_maxIter=cv_maxIter)
]

comparison_sdf = spark.createDataFrame(comparison_data)
comparison_sdf.show(truncate=False)
