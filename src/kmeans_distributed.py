import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint
np.random.seed(21)

from pyspark.sql import SparkSession
from pyspark.sql.types import *

import pyspark
conf = pyspark.SparkConf()
conf.set('spark.executor.memory', '2g')
spark = SparkSession.builder.appName('kmeans').getOrCreate()


def compute_new_centers():
    query = '''
        SELECT 
            AVG(x) AS x, 
            AVG(y) AS y,
            cluster_id AS id
        FROM points
        GROUP BY id
    '''
    centers = spark.sql(query)
    centers.createOrReplaceTempView('centers')
    centers.cache()


def compute_distances():
    query = '''
        SELECT
            p.x AS x,
            p.y AS y,
            c.id AS cluster_id,
            SQRT( POWER(p.x - c.x, 2) + 
                  POWER(p.y - c.y, 2) ) AS distance
        FROM points as p
        CROSS JOIN centers as c
    '''
    distances = spark.sql(query)
    distances.createOrReplaceTempView('distances')
    distances.cache()


def compute_new_clustering():
    query = '''
        WITH closest AS (
            SELECT
                x, y,
                MIN(distance) AS min_dist
            FROM distances
            GROUP BY x, y )

        SELECT
            sdf1.x AS x, 
            sdf1.y AS y, 
            sdf1.cluster_id AS cluster_id
        FROM distances AS sdf1
        INNER JOIN closest AS sdf2
        ON sdf1.x = sdf2.x
        AND sdf1.y = sdf2.y
        AND sdf1.distance = sdf2.min_dist
    '''    
    
    points = spark.sql(query)
    points.createOrReplaceTempView('points')
    points.cache()
    

def create_centers_sdf(centers):   
    field_x = StructField('x', FloatType(), True)
    field_y = StructField('y', FloatType(), True)
    field_id = StructField('id', IntegerType(), True)
    schema = StructType([field_x, field_y, field_id])
    return spark.createDataFrame(centers, schema)
    

def init_centers(X, Y, k):
    inds = randint(0, len(data) - 1, k)
    centers = []
    for i, j in enumerate(inds):
        x, y = X[j], Y[j]
        centers += {'x': x, 'y': y, 'id': i}
    return create_centers_sdf(centers)
    
    
def create_points_sdf(X, Y):
    field_x = StructField('x', FloatType(), True)
    field_y = StructField('y', FloatType(), True)
    field_id = StructField('id', IntegerType(), True)
    data = [{'x': x, 'y': y, 'id': 1} for x, y in zip(X, Y)]
    schema = StructType([field_x, field_y, field_id])
    return spark.createDataFrame(data, schema)
    

def kmeans_loop():
    compute_distances()
    compute_new_clustering()
    compute_new_centers()


def kmeans_distributed(X, Y, k=5, epochs=1):
    points = create_points_sdf(X, Y)
    points.createOrReplaceTempView('points')
    centers = init_centers(X, Y, k)
    centers.createOrReplaceTempView('centers')
    for _ in range(epochs):
        kmeans_loop()
        
        
def get_spark_clustering():
    clustering = spark.sql('SELECT * FROM points')
    return clustering.toPandas()
