import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import matplotlib.pyplot as plt

#Import data
data=pd.read_csv('data.csv')
#Identify independent variables values 
X=data.drop('',axis=1)
#Identtify dependent variables values
y=data.nameoftarget
#Split train-test data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
#Def grid search CV func

def gscv(model, xtrain, ytrain, xtest, ytest, lr, n, md, mcw, gamma, alpha, lmda, ss, csbt):
    #learning rate
    #n_estimators
    #max_depth
    #min_child_weight
    #gamma
    #reg_alpha
    #reg_lambda
    #subsample
    #colsample_bytree
    par={'learning_rate':[i/10 for i in range(lr[0],lr[1],lr[2])],'n_estimators':range(n[0],n[1],n[2]),'max_depth':range(md[0],md[1],md[2]),
    'min_child_weight':range(mcw[0],mcw[1],mcw[2]),'gamma':[i/10 for i in range(gamma[0],gamma[1],gamma[2])],'reg_alpha':[i/10 for i in range(alpha[0],alpha[1],alpha[2])],
    'reg_lambda':[i/10 for i in range(lmda[0],lmda[1],lmda[2])],'subsample':[i/10 for i in range(ss[0],ss[1],ss[2])],'colsample_bytree':[i/10 for i in range(csbt[0],csbt[1],csbt[2])]}
    grid_search=GridSearchCV(estimator = model, param_grid = par, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
    grid_search.fit(X_train,y_train)
    grid_search.best_params_, grid_search.best_score_
    print ('R2 Score on Train: ',metrics.r2_score(ytrain,grid_search.predict(xtrain)))
    print ('R2 Score on Test: ',metrics.r2_score(ytest,grid_search.predict(xtest)))

#Initiate a model and fixing learning rate and n_estimators
xgb_reg= xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=1000, silent=1, objective='reg:linear', 
booster='gbtree', n_jobs=1, gamma=0, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0, 
reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, missing=None, importance_type='gain')
#Tunning Paramaters
lr_1=[0,0,0]
n_1=[0,0,0]
md_1=[0,0,0]
mcw_1=[0,0,0]
gamma_1=[0,0,0]
alpha_1=[0,0,0]
lmda_1=[0,0,0]
ss_1=[0,0,0]
csbt_1=[0,0,0]

gscv(xgb_reg,X_train,y_train,X_test,y_test, lr_1,n_1,md_1,mcw_1,gamma_1,alpha_1,lmda_1,ss_1,csbt_1)
#Stochastic Optimization

