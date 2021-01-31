import numpy as np
import json
import argparse
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train_file', type=str, default='train.json')
parser.add_argument('-i', '--test_file', type=str, default='test.json')

args = parser.parse_args()
train_file = args.train_file
test_file = args.test_file


with open(train_file,'r') as f:
  train_data = json.load(f)

with open(test_file,'r') as f:
  test_data = json.load(f)
  
# create our training data from the tweets
X = [x["data"] for x in train_data]
# index all the sentiment labels
y = np.asarray([x["label"] for x in train_data])
X_test = [x for x in test_data]
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))), 
                     ('tfidf', TfidfTransformer(use_idf=True)), 
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',random_state=0,alpha=1e-4))])

parameters = {'vect__min_df': range(4, 6), 'vect__max_df': [0.4, 0.45, 0.5]}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
text_clf = gs_clf.fit(X, y)
best_param = gs_clf.best_params_
print(best_param)
text_clf = Pipeline([('vect', CountVectorizer(max_df=best_param['vect__max_df'],ngram_range=(1,2),min_df=best_param['vect__min_df'])), 
                     ('tfidf', TfidfTransformer(use_idf=True)), 
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',random_state=0,alpha=1e-4))])

text_clf = text_clf.fit(X, y)
predicted = text_clf.predict(X_test)
with open('output.txt', 'w') as f:
  for i in predicted:
      f.writelines(str(i))
      f.write('\n')
