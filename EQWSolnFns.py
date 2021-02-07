#!/usr/bin/env python
# coding: utf-8

print("""

##################################
EQWorks Solution
 
Author: Tryambak Kaushik
Date: 07 February 2021
##################################
""")
# In[3]:


# import findspark


# In[4]:


# findspark.init()


# In[5]:

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql import Row, Window

from datetime import date


# In[6]:


import pandas as pd
import math
import numpy as np
import os
from pprint import pprint


# In[8]:


def mysolution(sc,spark):
	# Load data/DataSample.csv to Spark DataFrame

	df_dataSample = spark.read.option("header",True).csv("data\DataSample.csv")
	print('\nDisplay Schema of DataSample.csv dataset table\n')
	df_dataSample.printSchema()


	# In[9]:


	#Display the contents of DataSample data
	print('\nDisplay contents of DataSample.csv dataset table\n')
	df_dataSample.show()


	# ### 1. Cleanup
	# 
	# A sample dataset of request logs is given in data/DataSample.csv. We consider records that have identical geoinfo and timest as suspicious. Please clean up the sample dataset by filtering out those suspicious request records.

	# In[22]:


	# Drop duplicate rows based on columns TimeSt, Latitude and Longitude

	df_clean = df_dataSample.dropDuplicates(['Latitude', 'Longitude']).dropDuplicates([' TimeSt'])
	print ('\nDisplay clean dataset after dropping suspicius requests (i.e., duplicate geoinfo and timest)\n')
	df_clean.show()

	print ("\n------------------------END OF ANSWER #1------------------------\n")
	# **End of Answer #1**
	# 
	# ---

	# ### 2. Label
	# Assign each request (from data/DataSample.csv) to the closest (i.e. minimum distance) POI (from data/POIList.csv).
	# 
	# **Note:** A POI is a geographical Point of Interest.

	# In[23]:


	#Load data from data/POIList.csv in Spark Dataframe

	df_poil = spark.read.option("header",True).csv("data\POIList.csv")
	
	print ('\nDisplay Schema and data of POIList dataset table\n')
	df_poil.printSchema()
	df_poil.show(5)


	# In[24]:


	#Convert pois Spark DataFrame to Pandas Dataframe
	df_pd_pois = df_poil.toPandas()


	# In[69]:


	#Python-UDF to find POI with minimum distance to each entry of DataSample
	def myfun(la2, lo2):
		
		min_dis = 1.0e10
		poi_id = df_pd_pois.loc[0,'POIID']
		
		for i, (la1,lo1) in enumerate( zip(df_pd_pois[' Latitude'], df_pd_pois['Longitude'])):
			la1, lo1 = float(la1), float(lo1)
			dis = math.sqrt((la1-la2)**2 + (lo1-lo2)**2)
			if min_dis > dis:
				min_dis = dis
				poi_id = df_pd_pois.loc[i,'POIID']
				
		return ([poi_id, min_dis])

	#Register Python-UDF with Spark-UDF
	myfun_spark = F.udf(myfun, ArrayType(StringType()))

	df_poi = df_clean.withColumn('temp_col', myfun_spark(  F.col('Latitude').cast(FloatType()),
											  F.col('Longitude').cast(FloatType())  )).cache()\
				.withColumn('POI', F.col('temp_col')[0])\
				.withColumn('POI_DIS', F.col('temp_col')[1].cast(DoubleType()))\
				.drop('temp_col')

	print('Display the dataframe with new columns of nearest POI and POI_DIS(i.e, distance to POI from request)')
	df_poi.show(5)

	print ("\n------------------------END OF ANSWER #2------------------------\n")
	# **End of Answer #2**
	# 
	# ---

	# ### 3. Analysis
	# For each POI, calculate the average and standard deviation of the distance between the POI to each of its assigned requests.
	# 
	# At each POI, draw a circle (with the center at the POI) that includes all of its assigned requests. Calculate the radius and density (requests/area) for each POI.

	# In[70]:


	#Group the dataframe df_poi on 'POI' column and calculate average and standard deviation on each group
	df_avgSD = df_poi.groupby('POI').agg(F.avg('POI_DIS').alias('Average'), F.stddev('POI_DIS').alias('Std_Dev'))

	#Left Join df_avgSD dataframe to df_poil dataframe for completeness
	df_avgSD = df_poil.join(df_avgSD, df_poil.POIID == df_avgSD.POI, how = 'Left').drop(df_avgSD.POI)

	print('Display distance Average and Std_Dev for each POI')
	df_avgSD.show()

	print ("Note: Based on above output, it can be concluded that POI2 radius of influence is ZERO\n")
	# **Note:** Based on above output, it can be concluded that POI2 radius of influence is ZERO

	# In[71]:


	#The radius of Influence-Circle of POI will be the distance to farthest assigned request

	w = Window.partitionBy('POI')

	df_radius = df_poi.withColumn('max_r', F.max('POI_DIS').over(w))                  .where(F.col('POI_DIS') == F.col('max_r'))                  .drop('max_r')

	#Left Join df_radius dataframe to df_poil dataframe for completeness
	df_avgSD_r = df_avgSD.join(df_radius['POI', 'POI_DIS'], df_avgSD.POIID == df_radius.POI, how = 'Left')                   .drop(df_radius.POI)                   .withColumnRenamed('POI_DIS', 'POI_RADIUS')

	print('Display the maximum POI_DIS (i.e, POI_RADIUS) values for each group\n')
	df_avgSD_r.show()


	# In[72]:


	#Calculate number of requests for each POI
	df_no_of_req = df_poi.groupby('POI').agg(F.count('POI').alias('Requests'))

	#Append POI_No.
	df_poi_req = df_avgSD_r.join(df_no_of_req, df_avgSD_r.POIID == df_no_of_req.POI, 'Left' )                         .drop(df_no_of_req['POI'])

	#Calculate the density
	df_poi_density = df_poi_req.withColumn('Density', F.col('Requests')/ (3.14*F.col('POI_RADIUS')**2 ))

	print('Dislay No. of Requests and Density for each POI')
	df_poi_density.show()

	print ("\n------------------------END OF ANSWER #3------------------------\n")
	# **End of Answer #3**
	# 
	# ---

	# ### 4. Data Science/Engineering Tracks
	# Please complete either 4a or 4b. Extra points will be awarded for completing both tasks.
	# 
	# #### 4a. Model
	# To visualize the popularity of each POI, they need to be mapped to a scale that ranges from -10 to 10. Please provide a mathematical model to implement this, taking into consideration of extreme cases and outliers. Aim to be more sensitive around the average and provide as much visual differentiability as possible.
	# Bonus: Try to come up with some reasonable hypotheses regarding POIs, state all assumptions, testing steps and conclusions. Include this as a text file (with a name bonus) in your final submission.

	# In[61]:


	#Import PySpark Libraries for Data Analytics
	from pyspark.ml.feature import MinMaxScaler
	from pyspark.ml.feature import VectorAssembler
	from pyspark.ml import Pipeline


	# In[88]:


	df_poi_density_temp = df_poi_density.filter(df_poi_density.Density.isNotNull())
	#df_poi_density_temp.show()


	# In[109]:


	# Spark-udf for converting column from vector type to double type
	myfun_vec2double = F.udf(lambda x: round(float(list(x)[0]),3), DoubleType())

	# Use Spark VectorAssembler Transformation - Converting column to vector type
	assembler = VectorAssembler(inputCols=['Density'],outputCol="Density_Vector")

	# Use Spark MinMaxScaler Transformation to scale the column within (min,max) range
	scaler = MinMaxScaler(min = -10, max = 10, inputCol="Density_Vector", outputCol="Density_Scaled")

	# Create a Spark Pipeline of VectorAssembler and MinMaxScaler
	pipeline = Pipeline(stages=[assembler, scaler])

	#Drop POI2 as outlier 
	df_poi_density_temp = df_poi_density.filter(df_poi_density.Density.isNotNull())

	# Spark fitting pipeline on dataframe
	df_norm = pipeline.fit(df_poi_density_temp).transform(df_poi_density_temp).withColumn("Density_Scaled", myfun_vec2double("Density_Scaled")).drop("Density_Vector")

	print('Display scaled density for each POI')
	df_norm.select(*['POIID'], *[F.round(c, 3).alias(c) for c in df_norm.columns[1:] ]).show()


	# In[112]:


	df_lognorm = df_norm.withColumn('log_Density', F.log10(F.col('Density')) )

	# Use Spark VectorAssembler Transformation - Converting column to vector type
	assembler_log = VectorAssembler(inputCols=['log_Density'],outputCol="log_Density_Vector")

	# Use Spark MinMaxScaler Transformation to scale the column within (min,max) range
	scaler_log = MinMaxScaler(min = -1.0, max = 1.0, inputCol="log_Density_Vector", outputCol="log_Density_Scaled")

	# Create a Spark Pipeline of VectorAssembler and MinMaxScaler
	pipeline_log = Pipeline(stages=[assembler_log, scaler_log])


	# Spark fitting pipeline on dataframe
	df_lognorm = pipeline_log.fit(df_lognorm).transform(df_lognorm)                  .withColumn("log_Density_Scaled", myfun_vec2double("log_Density_Scaled"))                  .drop("log_Density_Vector")

	print('Display scaled log_density for each POI')
	df_lognorm.select(*['POIID'], *[F.round(c, 3).alias(c) for c in df_lognorm.columns[1:] ]).show()


	#Save the interpretation on results in 'bonus' file
	bonus = """
	Interpretation:
	Density column is the ratio of Requests to POI_Area. log_Density was calculated by taking log10 of Density values. log_Density were scaled in range (-10,10) to calculate log_Density_Scaled.

	It is difficult to come up with a statitics with only 3 good POIs.

	Nonetheless, the density values of POI1 and POI3 are 3 orders higher than POI4. Hence, Density_Scaled, log_Density and log_Density_Scaled values are also skewed.
	POI1 and POI3 attract more customers or requests per unit area of influence.

	Assumptions: POI2 was dropped as outlier. POI2 data must be investigated to identify the cause of zero zone of influence. Bad data collection and formatting can be reasons for POI2 being outlier
	"""

	with open('bonus', 'w') as f:
		f.write(bonus)

	f.close()

	# 
	# **Interpretation:**
	# Density column is the ratio of Requests to POI_Area. log_Density was calculated by taking log10 of Density values. log_Density were scaled in range (-10,10) to calculate log_Density_Scaled.
	# 
	# It is difficult to come up with a statitics with only 3 good POIs.
	# 
	# Nonetheless, the density values of POI1 and POI3 are 3 orders higher than POI4. Hence, Density_Scaled, log_Density and log_Density_Scaled values are also skewed.
	# POI1 and POI3 attract more customers or requests per unit area of influence.
	# 
	# **Assumptions:** POI2 was dropped as outlier. POI2 data must be investigated to identify the cause of zero zone of influence. Bad data collection and formatting can be reasons for POI2 being outlier

	print ("\n------------------------END OF ANSWER #4a------------------------\n")
	# **End of Answer #4a**
	# 
	# ----

	# #### 4b. Pipeline Dependency
	# We use a modular design on all of our data analysis tasks. To get to a final product, we organize steps using a data pipeline. One task may require the output of one or multiple other tasks to run successfully. This creates dependencies between tasks.
	# 
	# We also require the pipeline to be flexible. This means a new task may enter a running pipeline anytime that may not have the tasks' dependencies satisfied. In this event, we may have a set of tasks already running or completed in the pipeline, and we will need to map out which tasks are prerequisites for the newest task so the pipeline can execute them in the correct order. For optimal pipeline execution, when we map out the necessary tasks required to execute the new task, we want to avoid scheduling tasks that have already been executed.
	# 
	# If we treat each task as a node and the dependencies between a pair of tasks as directed edges, we can construct a DAG (Wiki: Directed Acyclic Graph).
	# 
	# Consider the following scenario. At a certain stage of our data processing, we have a set of tasks (starting tasks) that we know all its prerequisite task has been executed, and we wish to reach to a later goal task. We need to map out a path that indicates the order of executions on tasks that finally leads to the goal task. We are looking for a solution that satisfies both necessity and sufficiency -- if a task is not a prerequisite task of goal, or its task is a prerequisite task for starting tasks (already been executed), then it shouldn't be included in the path. The path needs to follow a correct topological ordering of the DAG, hence a task needs to be placed behind all its necessary prerequisite tasks in the path.
	# 
	# Note: A starting task should be included in the path, if and only if it's a prerequisite of the goal task
	# 
	# For example, we have 6 tasks [A, B, C, D, E, F], C depends on A (denoted as A->C), B->C, C->E, E->F. A new job has at least 2 tasks and at most 6 tasks, each task can only appear once.
	# 
	# Examples:
	# 
	# Inputs: starting task: A, goal task: F, output: A,B,C,E,F or B,A,C,E,F.
	# Input: starting task: A,C, goal task:'F', outputs: C,E,F.
	# You will find the starting task and the goal task in question.txt file, list of all tasks in task_ids.txt and dependencies in relations.txt.
	# 
	# Please submit your implementation and result.

	# In[113]:


	#Assign questions data
	questions = {'starting task': '73', 'goal task': '36'}
	questions


	# In[114]:


	#Assign relations data
	relations = [(97,102),
				 (75,31),
				 (75,37),
				 (100,20),
				 (102,36),
				 (102,37),
				 (102,31),
				 (16,37),
				 (39,73),
				 (39,100),
				 (41,73),
				 (41,112),
				 (62,55),
				 (112,97),
				 (20,94),
				 (20,97),
				 (21,20),
				 (73,20),
				 (56,102),
				 (56,75),
				 (56,55),
				 (55,31),
				 (55,37),
				 (94,56),
				 (94,102)]


	# In[117]:


	#Assign Task-IDs data
	task_ids = [97,75,100,102,16,39,41,62,112,20,21,73,56,55,36,37,94,31]


	# In[118]:


	#Create a pandas-dataframe of relations data
	r_pd = pd.DataFrame(relations, columns = ['from', 'to'])
	#r_pd.head()


	# In[119]:


	#Get starting target (st) and goal target (gt)
	st = int(questions['starting task']); print ('Starting Task: %2d'%(st))
	gt = int(questions['goal task']); print ('Goal Task: %2d'%(gt))


	# In[171]:


	#A python recursive function to find the path from source to target
	def replicate_recur(st, gt, mylist=None):

		# If a list has not been passed as argument create an empty one
		if(mylist == None):
			mylist = [st]
			
		if st == gt:
			return mylist
		
		temp = r_pd[r_pd['from'] == st].values

		if not temp.any() :
			temp = 'Error'
			mylist.append(temp)
			return mylist
		
		mylist = [ [i for i in mylist] for _ in range(len(temp))]
		for idx,val in enumerate(temp[:,1]):
			mylist[idx].append(val)
			mylist[idx] = replicate_recur(val, gt, mylist[idx])

		return mylist

	output = []
	def removeNestings(l): 
		for i in l: 
			if (type(i) == list) & (type(i[0]) == list):
				removeNestings(i) 
			elif ('Error' not in i):
				output.append(i)

	print ('\nThe different paths from Starting Target to Goal Target\n')
	removeNestings([replicate_recur(st, gt)])
	pprint(output)

	print ("\n------------------------END OF ANSWER #4b------------------------\n")
	# In[ ]:


# In[7]:


#Create a SparkContext and SparkSession
# sc = SparkContext("local","firstapp")
# sc.stop()

# spark = SparkSession.builder.master("local").appName("EQ Soln").config("spark.some.config.option", "some-value").getOrCreate()
if __name__ == "__main__":
 
  # create Spark context with Spark configuration
  conf = SparkConf().setAppName("EQ Solution - Python").set("spark.hadoop.yarn.resourcemanager.address", "192.168.0.104:8032")
  sc = SparkContext(conf=conf)
  
# sc.stop()

  spark = SparkSession.builder.master("local").appName("EQ Soln").config("spark.some.config.option", "some-value").getOrCreate()

 
  mysolution(sc,spark)
