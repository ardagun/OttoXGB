import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold

df=pd.read_csv('trainotto.csv')
X= df.drop('target',1)
y=df['target']
X_encoded= pd.get_dummies(X, prefix_sep='_')
y_encoded= LabelEncoder().fit_transform(y)
X_encoded['target']= y_encoded
prediction_df= pd.read_csv('testotto.csv')
test_X=prediction_df.drop('id', 1)
test_X = preprocessing.StandardScaler().fit(test_X).transform(test_X.astype(float))

X_encoded.corr()['target'].abs().nlargest(15)
aa=X_encoded.corr()['target'].abs().nlargest(15)
X_highest= X_encoded.corr()['target'].abs().nlargest(15).index
plt.figure(figsize= (10,10), dpi=300)
sns.heatmap(X_encoded[X_highest].corr().abs())


X_encoded=X_encoded.set_index(X_encoded['id'])
X_encoded1=X_encoded.drop('id',1).drop('target',1)
X_scaled = preprocessing.StandardScaler().fit(X_encoded1).transform(X_encoded1.astype(float))

yyyy=y.unique()

kf=StratifiedKFold(n_splits=16,shuffle=True)  

for train, test in kf.split(X_scaled,y_encoded):
      X_train= X_scaled[train]
      y_train= y_encoded[train]
      X_val=X_scaled[test]  
      y_test=y_encoded[test]
     

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_val.shape,  y_test.shape)

clf_xgb= xgb.XGBClassifier(objective= 'multi:softmax', missing=None,
                           gamma=0.45,reg_alpha=3 ,reg_lambda=1, max_depth=7,n_estimators=500 
                           ,colsample_bytree=0.7)

clf_xgb.fit(X_train, y_train)

clf_predict=clf_xgb.predict(X_val)
pred_RF = clf_xgb.predict_proba(X_val)

print('Accuracy of xgb classifier on training set: {:.2f}'
     .format(clf_xgb.score(X_train, y_train)))
print('Accuracy of xgb classifier on test set: {:.2f}'
     .format(clf_xgb.score(X_val, y_test)))
print(accuracy_score(y_test,clf_predict))
print(classification_report(y_test, clf_predict))

from sklearn.metrics import log_loss
bb=log_loss(y_test, pred_RF)
print('Log Loss xgb: ', bb)
from sklearn.metrics import f1_score
ccc=f1_score(y_test, clf_predict, average='weighted')
print('F1 score: ',ccc)
