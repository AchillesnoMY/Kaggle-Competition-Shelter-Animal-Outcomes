import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

train_cat = pd.read_csv('train_cat.csv')
train_cat = train_cat.drop(['AnimalID'], axis=1)
label_cat = train_cat['OutcomeType']
train_cat = train_cat.drop(['OutcomeType'], axis=1)

train_dog = pd.read_csv('train_dog.csv')
train_dog = train_dog.drop(['AnimalID'], axis=1)
label_dog = train_dog['OutcomeType']
train_dog = train_dog.drop(['OutcomeType'], axis=1)


#tuning the parameters of gradientBoosting method for both dog and cat datasets.
def gradBosting_parameterTuning(train_dog,label_dog,train_cat,label_cat):
    #tuning the n_estimators and learning_rate first
    seed=123
    estimators_chosen=[800,1000]
    learning_chosen=[0.01,0.05]
    min_samples_split=[50,80,100]
    subsample=[0.4,0.6,0.8]
    max_depth=[8,10,12]
    min_score=np.inf
    best_params_dog={'n_estimators':0,'learning_rate':0,'min_samples_split':0,'subsample':0,'max_depth':0}

    gbc=GradientBoostingClassifier(max_features='sqrt',max_depth=6,min_samples_split=50,subsample=0.8)
    #tuning the estimators number and learning rate for dog
    for i in estimators_chosen:
        for j in learning_chosen:
            for z in min_samples_split:
                for m in subsample:
                    for n in max_depth:
                        gbc.set_params(n_estimators=i,learning_rate=j,min_samples_split=z,subsample=m,max_depth=n)
                        kfold=KFold(n_splits=10,random_state=seed)
                        score=cross_val_score(gbc,X=train_dog,y=label_dog,scoring='neg_log_loss',cv=kfold)
                        score=(-score.mean())
                        print('For dog dataset:')
                        print('The n_estimators=%d, the learning_rate=%.3f, the min_samples_split=%d, the subsample=%.2f,the max_depth=%d\
                             give the score=%f'%(i,j,z,m,n,score))
                        if score<min_score:
                            min_score=score
                            best_params_dog['n_estimators']=i
                            best_params_dog['learning_rate']=j
                            best_params_dog['min_samples_split']=z
                            best_params_dog['subsample']=m
                            best_params_dog['max_depth']=n
        print('Best params: {} {} {} {} {}, score: {}'.format(best_params_dog['n_estimators'], best_params_dog['learning_rate'],
                                                              best_params_dog['min_samples_split'],best_params_dog['subsample'],
                                                              best_params_dog['max_depth'],min_score))
    #tunning the estimators number and learning rate for cat
    min_score = np.inf
    best_params_cat = {'n_estimators': 0, 'learning_rate': 0, 'min_samples_split': 0, 'subsample': 0, 'max_depth': 0}
    for i in estimators_chosen:
        for j in learning_chosen:
            for z in min_samples_split:
                for m in subsample:
                    for n in max_depth:
                        gbc.set_params(n_estimators=i,learning_rate=j,min_samples_split=z,subsample=m,max_depth=n)
                        kfold=KFold(n_splits=10,random_state=seed)
                        score=cross_val_score(gbc,X=train_cat,y=label_cat,scoring='neg_log_loss',cv=kfold)
                        score=(-score.mean())
                        print('For cat dataset:')
                        print('The n_estimators=%d, the learning_rate=%.3f, the min_samples_split=%d, the subsample=%.2f,the max_depth=%d\
                              give the score=%f'%(i,j,z,m,n,score))
                        if score<min_score:
                            min_score=score
                            best_params_cat['n_estimators']=i
                            best_params_cat['learning_rate']=j
                            best_params_cat['min_samples_split']=z
                            best_params_cat['subsample']=m
                            best_params_cat['max_depth']=n
        print('Best params: {} {} {} {} {}, score: {}'.format(best_params_cat['n_estimators'], best_params_cat['learning_rate'],
                                                              best_params_cat['min_samples_split'],best_params_cat['subsample'],
                                                              best_params_cat['max_depth'],min_score))


def removeUnimporantFeat(train,test,gbc,type):
    feature_importance=pd.Series(gbc.feature_importances_,index=train.columns.values)
    feature_importance.sort_values(ascending=False,inplace=True)
    removed_features=None
    #print(feature_importance)
    if type=='dog':
        removed_features=feature_importance.index.values[feature_importance.values<0.02]
    if type=='cat':
        removed_features = feature_importance.index.values[feature_importance.values < 0.01]
    #print(removed_features)
    train=train.drop(removed_features,axis=1)
    test=test.drop(removed_features,axis=1)
    #print(train.columns)
    return train,test

def predictSeparately():
    seed=123
    # dog and cat datasets
    train_cat=pd.read_csv('train_cat.csv')
    train_cat=train_cat.drop(['AnimalID'],axis=1)
    label_cat=train_cat['OutcomeType']
    train_cat=train_cat.drop(['OutcomeType'],axis=1)

    train_dog=pd.read_csv('train_dog.csv')
    train_dog=train_dog.drop(['AnimalID'],axis=1)
    label_dog=train_dog['OutcomeType']
    train_dog=train_dog.drop(['OutcomeType'],axis=1)

    test_cat=pd.read_csv('test_cat.csv')
    id_cat=test_cat.ID
    test_cat=test_cat.drop(['ID'],axis=1)
    test_dog=pd.read_csv('test_dog.csv')
    id_dog=test_dog.ID
    test_dog=test_dog.drop(['ID'],axis=1)

    gbc_dog=GradientBoostingClassifier(n_estimators=800,max_features='sqrt',max_depth=8,min_samples_split=50,subsample=0.8,
                                      learning_rate=0.01)
    gbc_cat=GradientBoostingClassifier(n_estimators=1000,max_features='sqrt',max_depth=8,min_samples_split=50,subsample=0.8,
                                      learning_rate=0.01)

    gbc_dog.fit(train_dog,label_dog)
    train_dog,test_dog=removeUnimporantFeat(train_dog,test_dog,gbc_dog,'dog')
    # cross-validation for dog
    kfold = KFold(n_splits=10, random_state=seed)
    score = cross_val_score(gbc_dog, train_dog, label_dog, scoring='neg_log_loss', cv=kfold)
    print('The cross_validation score for dog is:', -score.mean()) # 0.8975439759080256 0.8966528220538722
    gbc_dog.fit(train_dog,label_dog)
    predictions_dog=gbc_dog.predict_proba(test_dog)

    gbc_cat.fit(train_cat,label_cat)
    train_cat,test_cat=removeUnimporantFeat(train_cat,test_cat,gbc_cat,'cat')
    #cross-validation for cat
    kfold=KFold(n_splits=10,random_state=seed)
    score=cross_val_score(gbc_cat,train_cat,label_cat,scoring='neg_log_loss',cv=kfold)
    print('The cross_validation score for cat is:',-score.mean())# 0.48536095950197244 0.48400425507979766
    gbc_cat.fit(train_cat,label_cat)
    predictions_cat=gbc_cat.predict_proba(test_cat)

    columns=gbc_dog.classes_
    output_dog=pd.DataFrame(predictions_dog,columns=columns)
    output_dog=pd.concat([id_dog,output_dog],axis=1)

    output_cat=pd.DataFrame(predictions_cat,columns=columns)
    output_cat=pd.concat([id_cat,output_cat],axis=1)

    output_combination=pd.concat([output_cat,output_dog],axis=0)
    output_combination_gbc=output_combination.sort_values(by='ID',ascending=True)
    print(output_combination_gbc)
    output_combination_gbc.to_csv('output_comb_gbc.csv',index=False)
