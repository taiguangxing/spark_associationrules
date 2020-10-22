from pyspark.mllib.fpm import FPGrowth
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession
import itertools
import os
spark = SparkSession.builder.appName('asso').getOrCreate()
data = spark.sparkContext.textFile("D:\python_project\spark_associationrules\sample_fpgrowth.txt")
NUM_DATA = len(data.collect())
print('总数据量',NUM_DATA)
transactions = data.map(lambda line: line.strip().split(' '))
model = FPGrowth.train(transactions, minSupport=0.3, numPartitions=10)
result = model.freqItemsets()
tmp = result.collect()
for fi in tmp:
    print(fi)
print('频繁项集数量',len(tmp))
freqDict = result.map(lambda x:[tuple(sorted(x[0])), x[1]]).collectAsMap()
print(freqDict)

def subSet(listVariable):  # 求列表所有非空真子集的函数
    newList = []
    for i in range(1, len(listVariable)):
        newList.extend(list(itertools.combinations(listVariable, i)))
    return newList
def computeConfidence(freqItemset):
    itemset = freqItemset[0]
    freq = freqItemset[1]
    subItemset = subSet(itemset)
    rules = []
    for i in subItemset:
        complement = tuple(set(itemset).difference(set(i)))  # 取补集
        itemLink = str(complement) + '->' + str(i)
        confidence = float(freq) / freqDict[tuple(sorted(complement))]  # 求置信度
        lift = float(freq) * NUM_DATA / (freqDict[tuple(sorted(i))] * freqDict[tuple(sorted(complement))])
        antecedent  = str(complement) #前件
        consequent = str(i) #后件
        rule = [antecedent,consequent,itemLink, confidence,lift]
        rules.append(rule)
    return rules
confidence = result.flatMap(computeConfidence) # 使用flatMap可将map映射后的结果“摊平”
tmp_confidence = confidence.collect()
for i in tmp_confidence:
    print(i)
print('计算完成置信度，所有规则数目',len(tmp_confidence))
minSupportConfidence = confidence.filter(lambda x: x[3] > 0.5).filter(lambda x: x[4]>1)  #保留置信度大于0.5，支持度大于1的规则
print('过滤后规则数目',len(minSupportConfidence.collect()))
for rules in (minSupportConfidence.collect()):
    print(rules)