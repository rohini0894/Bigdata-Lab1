#!/usr/bin/env python
# coding: utf-8

# In[89]:


from pyspark.sql import SparkSession

spark=SparkSession.builder.appName("MyfisrtCSVLoad").getOrCreate()


# In[90]:


df = spark.read.csv("BostonHousing.csv",header=True)


# In[91]:


df.show(5)


# In[92]:


df.count()


# In[93]:


df.printSchema()


# In[94]:


df=df.drop("b")


# In[95]:


df.show()


# In[96]:


import pyspark.sql.functions as func
df = df.withColumn("crim", func.round(df["crim"], 2))
df = df.withColumn("nox", func.round(df["nox"], 2))
df = df.withColumn("rm", func.round(df["rm"], 2))
df = df.withColumn("dis", func.round(df["dis"], 2))
df.show()


# In[97]:


df= df.withColumn("Age10", func.round(df.age*0.1+df.age, 2))
df.show()


# In[98]:


df.printSchema()


# In[99]:


from pyspark.sql.types import *
dfplot = df.groupBy("Age10").count()
dfplot.show()


# In[116]:


from matplotlib import pyplot as plt
import pandas

x= dfplot.toPandas()["Age10"].values.tolist()
y = dfplot.toPandas()["count"].values.tolist()

plt.bar(x,y, color = "green")
plt.xlabel("Age10")
plt.ylabel("count")
plt.show()


# In[107]:


df.describe('crim', 'zn', 'indus', 'chas','nox').show()


# In[109]:


df.describe('dis','rad','tax','ptratio','lstat').show()


# In[110]:


df.describe('rm','age','medv','Age10').show()


# In[112]:


pandasDF = df.toPandas()
print(pandasDF)


# In[113]:


pandasDF.tail(5)


# In[ ]:




