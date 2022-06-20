import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import time
from datetime import timedelta

start_time = time.monotonic()

dat=pd.read_csv('thesisdata.csv')

dat_tc=dat.drop('visc[mPa.s]',axis=1)
dat_tc=dat_tc.dropna()
dat_v=dat.drop('tc-virial[W/m.K]',axis=1)
dat_v=dat_v.dropna()

X_tc = dat_tc.drop('tc-virial[W/m.K]', axis=1)
X_v = dat_v.drop('visc[mPa.s]', axis=1)

#Split data

xtc_train=X_tc[X_tc['alc']!='glycerol']
xtc_train=xtc_train.drop('alc', axis=1)
ytc_train=dat_tc[dat_tc['alc']!='glycerol']
ytc_train=ytc_train['tc-virial[W/m.K]']

xtc_test=X_tc[X_tc['alc']=='glycerol']
xtc_test=xtc_test.drop('alc', axis=1)
ytc_test=dat_tc[dat_tc['alc']=='glycerol']
ytc_test=ytc_test['tc-virial[W/m.K]']

xv_train=X_v[X_v['alc']!='glycerol']
xv_train=xv_train.drop('alc', axis=1)
yv_train=dat_v[dat_v['alc']!='glycerol']
yv_train=yv_train['visc[mPa.s]']

xv_test=X_v[X_v['alc']=='glycerol']
xv_test=xv_test.drop('alc', axis=1)
yv_test=dat_v[dat_v['alc']=='glycerol']
yv_test=yv_test['visc[mPa.s]']



#Def a grid search cv function n return best params
def gscv(model,xtrain,ytrain,xtest,ytest,lr,n,md,mcw,gamma,alpha,lmda,ss,csbt):
    par={'learning_rate':[i/10 for i in range(lr[0],lr[1],lr[2])],'n_estimators':range(n[0],n[1],n[2]),'max_depth':range(md[0],md[1],md[2]),'min_child_weight':range(mcw[0],mcw[1],mcw[2]),'gamma':[i/10 for i in range(gamma[0],gamma[1],gamma[2])],'reg_alpha':[i/10 for i in range(alpha[0],alpha[1],alpha[2])],'reg_lambda':[i/10 for i in range(lmda[0],lmda[1],lmda[2])],'subsample':[i/10 for i in range(ss[0],ss[1],ss[2])],'colsample_bytree':[i/10 for i in range(csbt[0],csbt[1],csbt[2])]}
    grid_search=GridSearchCV(estimator=model,param_grid=par,scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=8)
    grid_search.fit(xtrain,ytrain)
    print ('R2 score on train: ',metrics.r2_score(ytrain,grid_search.predict(xtrain)))
    print ('R2 score on test: ',metrics.r2_score(ytest,grid_search.predict(xtest)))
    return grid_search.best_params_

#Initiating a model
xgb_reg=xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=1000, silent=1, objective='reg:linear', booster='gbtree', n_jobs=1, gamma=0, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, missing=None, importance_type='gain')

#Parameters ranges
lr=[1,10,4]
n=[1,1001,100]
md=[1,10,4]
mcw=[1,10,4]
gamma=[1,10,4]
alpha=[1,10,4]
lmda=[1,10,4]
ss=[1,10,4]
csbt=[1,10,4]

#Tunning parameters 1
b_par_tc=gscv(xgb_reg,xtc_train,ytc_train,xtc_test,ytc_test,lr,n,md,mcw,gamma,alpha,lmda,ss,csbt)
b_par_v=gscv(xgb_reg,xv_train,yv_train,xv_test,yv_test,lr,n,md,mcw,gamma,alpha,lmda,ss,csbt)

end_time = time.monotonic()
optime = timedelta(seconds=end_time - start_time)
#Write on an external file
f= open('param.txt','w+')
f.write('TC model ')
f.write(str(b_par_tc))
f.write('Visc model ')
f.write(str(b_par_v))
f.write('Time \n')
f.write(str(optime))
f.close()
