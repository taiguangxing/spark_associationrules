from pyspark.ml.fpm import FPGrowth
from pyspark import SparkConf
from pyspark.sql import SparkSession
conf=SparkConf()
spark = SparkSession.builder.appName('asso').getOrCreate()
data = spark.sparkContext.textFile("D:\python_project\spark_associationrules\sample_fpgrowth.txt")
data = [(ind,list(data.split(' '))) for ind,data in enumerate(data.collect())]
df = spark.createDataFrame(data, ["id", "items"])
df.show()
fpGrowth = FPGrowth(itemsCol='items',minSupport=0.3,minConfidence=0.5)
model = fpGrowth.fit(df)
frequentItem = model.freqItemsets
ass_rules = model.associationRules
prediction = model.transform(df)
frequentItem.show()
print(frequentItem.count())
ass_rules.show()
prediction.show()
model.write().overwrite().save('D:\python_project\spark_associationrules\model')