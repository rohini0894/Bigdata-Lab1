#!/usr/bin/env python
# coding: utf-8

# In[10]:


#Prepare your google colab PySpark Session as you did before.
from pyspark.sql import SparkSession

spark=SparkSession.builder.appName("Module4").getOrCreate()

#Read 'BostonHousing.csv' dataset in PySpark. You may use other dataset as well if you want to.
house_df = spark.read.csv("BostonHousing.csv",inferSchema= True,header=True)

house_df.show(5)


# In[ ]:


#Part 1


# In[23]:


#Combine features to a single vector columns using VectorAssembler 
#(all columns other than target column 'medv')

#required lib
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[13]:


assembler = VectorAssembler(inputCols=['crim',
 'zn',
 'indus',
 'chas',
 'nox',
 'rm',
 'age',
 'dis',
 'rad',
 'tax',
 'ptratio',
 'b',
 'lstat'], outputCol="features")

output_df=assembler.transform(house_df)

output_df.show(5)


# In[14]:


feat_df=assembler.transform(house_df).select("medv","features")
feat_df.show(5)


# In[17]:


#Standardize features for principal component analysis

scaler = StandardScaler(
    inputCol="features",
    outputCol="ScaledFeatures",
    withMean = True,
    withStd = True
    )

scalerModel = scaler.fit(output_df)

df_scaled = scalerModel.transform(output_df)

df_scaled.select(["medv", "ScaledFeatures"]).show(5)



# In[20]:


scaled_df=df_scaled.select("features", "ScaledFeatures")
scaled_df.show(5)


# In[22]:


#Train model for two principal component analysis

n_components = 2

pca =PCA(
    k=n_components,
    inputCol = "ScaledFeatures",
    outputCol = "pcaFeatures").fit(scaled_df)

df_pca = pca.transform(scaled_df)

print("Explained Variance Ratio", pca.explainedVariance.toArray())

df_pca.show()


# In[33]:


#Collect distributed data in numpy array and then convert to pandas dataframe

cols = ['crim',
 'zn',
 'indus',
 'chas',
 'nox',
 'rm',
 'age',
 'dis',
 'rad',
 'tax',
 'ptratio',
 'b',
 'lstat']

pcs = np.round(pca.pc.toArray(),4)
pcs

df_pc = pd.DataFrame(pcs, columns = ['PC'+str(i) for i in range(1, 3)], index = cols)
df_pc


# In[39]:


#Plot two principal components

import seaborn as sb
import matplotlib.pyplot as plt
 
plt.figure(figsize = (6,6))
sb.scatterplot(data = df_pc, x = 'PC1',y = 'PC2')


# In[ ]:


#Part 2


# In[ ]:


house_df = spark.read.csv("BostonHousing.csv",inferSchema= True,header=True)

house_df.show(5)
house_df.printSchema()
house_df.describe().toPandas().transpose()


# In[ ]:


from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat'], outputCol = 'features')
vhouse_df = vectorAssembler.transform(house_df)
vhouse_df = vhouse_df.select(['features', 'medv'])
vhouse_df.show(3)


# In[ ]:


splits = vhouse_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]


# In[ ]:


from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol = 'features', labelCol='medv', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))

print("Intercept: " + str(lr_model.intercept))


# In[ ]:


trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
print("MSE: %f" % trainingSummary.meanSquaredError)
print("MAE: %f" % trainingSummary.meanAbsoluteError)


# In[ ]:


train_df.describe().show()


# In[ ]:


lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","medv","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="medv",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))


# In[ ]:


test_result = lr_model.evaluate(test_df)
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)


# In[ ]:


print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()


# In[ ]:


predictions = lr_model.transform(test_df)
predictions.select("prediction","medv","features").show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




