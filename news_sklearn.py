# -*- coding=utf-8 -*-
# @Time : 2021/8/20 16:11
# @Author : wangshuang
# @File : news_sklearn.py
# @Software : PyCharm
import re
import pandas as pd
import numpy as np


filename = 'data/20newsgroups.json'
categories = ['alt.atheism',
              'comp.graphics',
              'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware',
              'comp.windows.x',
              'misc.forsale',
              'rec.autos',
              'rec.motorcycles',
              'rec.sport.baseball',
              'rec.sport.hockey',
              'sci.crypt',
              'sci.electronics',
              'sci.med',
              'sci.space',
              'soc.religion.christian',
              'talk.politics.guns',
              'talk.politics.mideast',
              'talk.politics.misc',
              'talk.religion.misc']

def data_cols(data=categories):
    # org_df = pd.DataFrame(data, columns=['index', 'topic'])
    # org_df.to_csv('data/org_cols.csv', index=True)
    data = pd.read_json(filename, orient='records')
    cntByLabel = data.groupby("label").count()
    cols_info = pd.DataFrame({'index': range(20) , 'label': categories, 'cnt': cntByLabel.iloc[:, 0]})
data_cols()

def data_preprocess(filename=filename):
    data_1 = pd.read_json(filename, orient='records')
    pattern1 = '|'.join([',', ';', '_', '\*',  '\s+', '\{', '\}', '\:', '\#', '\$', '\%', '\.', '\!', '\(', '\)', '\?', '\<', '\>', '\|', '-'])
    words = pd.read_csv("data/stopwords_nlp.txt", sep='\n', header=None)
    stopwords = [word.strip() for word in words[0]]
    # stopwords.append('')
    res = []
    for i in range(data_1.shape[0]):
        news_1 = re.split(pattern1, data_1.loc[i, 'news']) #分割
        words_filtered = [word.lower() for word in news_1 if word.lower() not in stopwords and len(word) != 1] #105过滤
        res.append(words_filtered)
    data_1['words'] = res
    sdata = data_1['words'].apply(lambda x: ' '.join(x)) #pd.series
    return sdata, data_1['label']

data, label = data_preprocess()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


import numpy as np

vectorizer = TfidfVectorizer()
idf_matrix = vectorizer.fit_transform(data) #(2000, 41743)，该稀疏矩阵共计189141个非0元素,占比0。2%
vectorizer.vocabulary_ #168253 dict写入到res/vocabulary_sklearn_voc/idx
dim = idf_matrix.nonzero()

# pca = PCA(n_components=1000, svd_solver='arpack', tol=0.1) #TypeError: PCA does not support sparse input
pca = TruncatedSVD(n_components=2000, n_iter=10) #使用pca和lr的最佳效果77.45%, 82.5%
pca_matrix = pca.fit_transform(idf_matrix) #pca_matrix中会产生负值，不适合NB
pca.explained_variance_ratio_

X_tr, X_te, y_tr, y_te = train_test_split(idf_matrix, label, test_size=0.2, random_state=2021) #(16961, 1885)

# 测试性能
# estimator = MultinomialNB() #79.95%, 81.5%，不使用该模型
lr = LogisticRegression(C=10, multi_class='multinomial', max_iter=1000) ##81,1%, 84%, 0.01, 1, 10, 100(81.4%,86.5%), 10和100差不多
cv_res = cross_validate(lr, idf_matrix, label, cv=3)

#测试误差原因,理论推测应该集中在易混淆的类别
lr_model = lr.fit(X_tr, y_tr)
pre = lr_model.predict(X_te)
# pre1 = lr_model.predict(X_tr)

np.sum(y_te == 3) #63
tp, num = np.unique(pre[y_te == 3], return_counts=True)

def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))
show_top10(lr_model, vectorizer, categories)

res = str(classification_report(y_tr, pre, output_dict=False))
# res_df = pd.DataFrame(res_dict)
with open("res/sklearn/sklearn_report_tr.txt", 'w') as f:
    f.write(res)
    f.close()


cv_res = cross_validate(lr, pca_matrix, label, cv=10) #使用pca降维，提高效率，但是可能损失accuracy


# pre = lr.predict(pca_matrix)
# print(classification_report(label, pre))


mean, best = np.mean(cv_res['test_score']), np.max(cv_res['test_score'])

#网格搜索与可视化
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

lr = LogisticRegression(max_iter=10000,  multi_class='multinomial')
param_grid ={
    'lr__C': [100, 300, 500]
    #np.logspace(-3, 3, 6)
}
pipe = Pipeline(steps=[('lr', lr)])
search = GridSearchCV(pipe, param_grid, cv=5)
search.fit(idf_matrix, label)
search.cv_results_

with open('data/grid_cv.txt', 'a') as log:
    print(search.cv_results_, file=log)
    log.close()

import matplotlib.pyplot as plt

print(search.cv_results_)
res = pd.DataFrame(search.cv_results_)
fig, ax = plt.subplots(figsize=(6, 6))
res.plot(x='param_lr__C', y='mean_test_score', yerr='std_test_score', legend=False, ax=ax)

ax.set_xlabel('lr_C')
ax.set_ylabel('mean_test_score')
# plt.ylim(0.9, 1)
plt.show()

s = "abc 123 456 { } 567"
pattern = '|'.join(
    [',', ';', '_', '\*', '\s+', '\{', '\}', '\:', '\#', '\$', '\%', '\.', '\!', '\(', '\)', '\?', '\<', '\>', '\|',
     '-'])




