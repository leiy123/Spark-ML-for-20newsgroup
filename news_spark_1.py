# -*- coding=utf-8 -*-
# @Time : 2021/9/3 15:22
# @Author : wangshuang
# @File : news_spark.py
# @Software : PyCharm

import pandas as pd
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, functions
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, CountVectorizer, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

filename = 'data/20newsgroups.json'
# config spark env
spark = SparkSession.builder.master("local[*]").config("spark.driver.memory", "8g").config("spark.executor.memory", "8g").appName("news_analyze").getOrCreate()

def preProcess(filename=filename):
    #data load
    df = spark.read.json(filename)
    # df.createOrReplaceTempView('news')
    # sql_str = "select label, count(1) as num from news group by label order by label"
    # label_num = spark.sql(sql_str)
    # label_num.show()
    #filter
    pattern = '|'.join(
        [',', ';', '_', '\*', '\\s+', '\{', '\}', '\:', '\#', '\$', '\%', '\.', '\!', '\(', '\)', '\?', '\<', '\>', '\|',
         '-'])

    regexTokenizer = RegexTokenizer(inputCol="news", outputCol="words", toLowercase=True, pattern=pattern,
                                    minTokenLength=2)
    df = regexTokenizer.transform(df)

    # df.printSchema()
    # remove stopwords
    words = pd.read_csv("data/stopwords_nlp.txt", sep='\n', header=None)
    # stopwords = words[0].to_list()  # 891-github
    stopwords = [word.strip() for word in words[0]]
    remover = StopWordsRemover(inputCol='words', outputCol='filter', stopWords=stopwords)
    words_filtered = remover.transform(df)

    concat_func_1 = functions.udf(lambda words : ' '.join(words)) # words是列表
    # df1 = words_filtered.withColumn("csv_news", concat_func_1(words_filtered.filter))
    df1 = words_filtered.select(concat_func_1("filter").alias("csv_news"), "label")
    df1.count()


    regexTokenizer_1 = RegexTokenizer(inputCol="csv_news", outputCol="words_1", gaps=False, pattern="\\b\\w\\w+\\b")
    df2 = regexTokenizer_1.transform(df1)

    # hashtf = HashingTF(numFeatures=1 << 18, inputCol='filter', outputCol='words_tf')
    # words_tf = hashtf.transform(words_filtered)
    #CountVectorizer
    vector = CountVectorizer(inputCol='words_1', outputCol='words_tf').fit(df2) # vocabulary:242655
    words_tf = vector.transform(df2)
    # vector.save("countVector")
    idf = IDF(inputCol='words_tf', outputCol='words_idf') # minDocFreq=2
    words_idf = idf.fit(words_tf).transform(words_tf)
    vecAbler = VectorAssembler(inputCols=['words_tf', 'words_idf'], outputCol="features")
    data = vecAbler.transform(words_idf)
    return data.select(['features', 'label'])

def modelMake(data=None):
    data_tr, data_te = data.randomSplit([0.8, 0.2], 2021)
    # data_tr.count()
    # data_te.count()
    lr = LogisticRegression(regParam=0.1, elasticNetParam=0, family="multinomial", tol=1e-4, maxIter=1000)
    lrModel = lr.fit(data_tr)
    pre = lrModel.transform(data_te)
    # pre = lrModel.transform(data_tr)
    # pre.groupby("prediction").count().show()
    predictions = pre.select(col("label").cast("Double"), col("prediction"))
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    weigthPrecision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    weightedRecall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)
    return accuracy, weigthPrecision, weightedRecall, f1
    '''
    data.select(['features', 'label']).show(5)
    data_tr, data_te = data.randomSplit([0.8, 0.2], 2021) # 14177+4669=18846
    data_tr.count()
    data_te.count()
    data_tr.groupby(['label']).count().collect()
    lr = LogisticRegression(regParam=0.01, elasticNetParam=0, family="multinomial", tol=1e-4, maxIter=10000)
    lrModel = lr.fit(data_tr)
    pre = lrModel.transform(data_te)
    pre.take(3)[2].prediction
    for row in pre.take(10):
        print("pre:%f ---- label:%f" % (row.prediction, row.label))
    predictions = pre.select(col("label").cast("Float"), col("prediction"))
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    weigthPrecision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    weightedRecall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)
    '''
# lr model fit (regParam = 1 / C(sklearn), elasticNetParam in (0(L2), 1(L1)) = penalty(sklearn))
def modelTuning(data=None):
    lr = LogisticRegression(family='multinomial', tol=1e-4, maxIter=100)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    paramGrid = ParamGridBuilder().addGrid(lr.regParam, np.logspace(-3, 1, num=5, base=5)).build()
    cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5, parallelism=2)
    cvModel = cv.fit(data)
    return cvModel

if __name__ == '__main__':
    data = preProcess()
    accuracy, weigthPrecision, weightedRecall, f1 = modelMake(data)
    cvModel = modelTuning(data)
    # with open('data/spark_res.txt', 'a') as log:
    #     print(' '.join([filename, str(accuracy), str(weigthPrecision), str(weightedRecall), str(f1)]), file=log)
    #     log.close()
    print("down-----------")
# modelTuning(data)

# with open('res/spark/spark_voc.txt', 'w') as log:
#     print(' '.join(vector.vocabulary), file=log)
#     log.close()
# with open('res/vocabulary_spark.txt', 'a') as voc_log:
#     res = ' '.join(vector.vocabulary)
#     print(res, file=voc_log)
#     # print(' '.join([filename, str(accuracy), str(weigthPrecision), str(weightedRecall), str(f1)]), file=voc_log)
#     voc_log.close()



# Obtain the objective per iteration
# https://spark.apache.org/docs/latest/ml-classification-regression.html#linear-regression

# lrModel体现测试数据集已经被训练的特别好，基本上没有错误
# print("Coefficients: \n" + str(lrModel.coefficientMatrix))
# print("Intercept: " + str(lrModel.interceptVector))
#
# trainingSummary = lrModel.summary
#
# # Obtain the objective per iteration
# objectiveHistory = trainingSummary.objectiveHistory
# print("objectiveHistory:")
# for objective in objectiveHistory:
#     print(objective)
#
# # for multiclass, we can inspect metrics on a per-label basis
# print("False positive rate by label:")
# for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
#     print("label %d: %s" % (i, rate))
#
# print("True positive rate by label:")
# for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
#     print("label %d: %s" % (i, rate))
#
# print("Precision by label:")
# for i, prec in enumerate(trainingSummary.precisionByLabel):
#     print("label %d: %s" % (i, prec))
#
# print("Recall by label:")
# for i, rec in enumerate(trainingSummary.recallByLabel):
#     print("label %d: %s" % (i, rec))
#
# print("F-measure by label:")
# for i, f in enumerate(trainingSummary.fMeasureByLabel()):
#     print("label %d: %s" % (i, f))
#
# accuracy = trainingSummary.accuracy
# falsePositiveRate = trainingSummary.weightedFalsePositiveRate
# truePositiveRate = trainingSummary.weightedTruePositiveRate
# fMeasure = trainingSummary.weightedFMeasure()
# precision = trainingSummary.weightedPrecision
# recall = trainingSummary.weightedRecall
# print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
#       % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))

