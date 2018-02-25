import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV


def extraTree_parameterTuning(train,label):
    #tuning the n_estimators first
    seed=123
    cv_results=[]
    estimators_chosen=[20,100,500,800,1000]
    extraTree=ExtraTreesClassifier(n_estimators=20,oob_score=True,max_features='sqrt',max_depth=8,bootstrap=True)
    for i in estimators_chosen:
        extraTree.set_params(n_estimators=i)
        kfold=KFold(n_splits=10,random_state=seed)
        score=cross_val_score(extraTree,X=train,y=label,scoring='neg_log_loss',cv=kfold)
        cv_results.append(-score.mean())
    print('The cross-validation scores:',cv_results)
    sns.set(style='darkgrid')
    plt.plot(estimators_chosen,cv_results)
    plt.xlabel('estimators',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.show()

    #tuning the min_samples_leaf
    cv_results=[]
    minSamples_chosen = [2,4,6,8,10]
    extraTree=ExtraTreesClassifier(n_estimators=800,oob_score=True,max_features='sqrt',max_depth=8,bootstrap=True)
    for j in minSamples_chosen:
        extraTree.set_params(min_samples_leaf=j)
        kfold=KFold(n_splits=10,random_state=seed)
        score=cross_val_score(extraTree,X=train,y=label,scoring='neg_log_loss',cv=kfold)
        cv_results.append(-score.mean())
    print('The cross-validation scores:', cv_results)
    sns.set(style='darkgrid')
    plt.plot(minSamples_chosen,cv_results)
    plt.xlabel('estimators',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.show()
    #tuning the max_depth

    cv_results=[]
    maxDepth_chosen=[2,4,6,8,10]
    extraTree=ExtraTreesClassifier(n_estimators=500,oob_score=True,max_features='sqrt',max_depth=8,min_samples_leaf=2,bootstrap=True)
    for k in maxDepth_chosen:
        extraTree.set_params(max_depth=k)
        kfold=KFold(n_splits=10,random_state=seed)
        score=cross_val_score(extraTree,X=train,y=label,scoring='neg_log_loss',cv=kfold)
        cv_results.append(-score.mean())
    print('The cross-validation scores:', cv_results)
    sns.set(style='darkgrid')
    plt.plot(maxDepth_chosen,cv_results)
    plt.xlabel('estimators',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.show()

#plot feature importance
def plotFeatureImportance(et,train):
    feature_importance=pd.Series(et.feature_importances_,index=train.columns.values)
    feature_importance.sort_values(ascending=False,inplace=True)
    sns.set(style='darkgrid')
    feature_importance.plot(kind='bar')
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig('feature_importance_rf.png')
    plt.show()

def removeUnimporantFeat(train,test,et):
    feature_importance=pd.Series(et.feature_importances_,index=train.columns.values)
    feature_importance.sort_values(ascending=False,inplace=True)
    #print(feature_importance)
    removed_features=feature_importance.index.values[feature_importance.values<0.02]
    #print(removed_features)
    train=train.drop(removed_features,axis=1)
    test=test.drop(removed_features,axis=1)
    #print(train.columns)
    return train,test

seed=123
def predictSingle():
    train = pd.read_csv('newTrain.csv')
    train = train.drop(['AnimalID'], axis=1)
    label = train['OutcomeType']
    train = train.drop(['OutcomeType'], axis=1)
    test = pd.read_csv('newTest.csv')
    id = test.ID
    test = test.drop(['ID'], axis=1)

    et=ExtraTreesClassifier(n_estimators=800,max_features='sqrt',max_depth=10,min_samples_leaf=2,random_state=
                          seed)
    et.fit(train,label)
    plotFeatureImportance(et,train)
    train,test=removeUnimporantFeat(train,test,et)
    et.fit(train,label)
    #cross-validation for rf
    kfold=KFold(n_splits=10,random_state=seed)
    score=cross_val_score(et,train,label,scoring='neg_log_loss',cv=kfold)
    print(-score.mean())
    columns=et.classes_
    predictions=et.predict_proba(test)
    output_et=pd.DataFrame(predictions,columns=columns)
    output_et=pd.concat([id,output_et],axis=1)
    output_et.to_csv('output_et.csv',index=False)

def predictSeparately():
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

    et_dog=ExtraTreesClassifier(n_estimators=1000,max_features='sqrt',max_depth=12,min_samples_leaf=3,random_state=
                              seed)#0.9544635255505858 0.9495109526227502 0.9493983344568621
    et_cat=ExtraTreesClassifier(n_estimators=1000,max_features='sqrt',max_depth=12,min_samples_leaf=3,random_state=
                              seed)#0.5629159176576117 0.5497040454891714 0.5497040454891714

    et_dog.fit(train_dog,label_dog)
    train_dog,test_dog=removeUnimporantFeat(train_dog,test_dog,et_dog)
    # cross-validation for dog
    kfold = KFold(n_splits=10, random_state=seed)
    score = cross_val_score(et_dog, train_dog, label_dog, scoring='neg_log_loss', cv=kfold)
    print('The cross_validation score for dog is:', -score.mean())
    et_dog.fit(train_dog,label_dog)
    predictions_dog=et_dog.predict_proba(test_dog)

    et_cat.fit(train_cat,label_cat)
    train_cat,test_cat=removeUnimporantFeat(train_cat,test_cat,et_cat)
    #cross-validation for cat
    kfold=KFold(n_splits=10,random_state=seed)
    score=cross_val_score(et_cat,train_cat,label_cat,scoring='neg_log_loss',cv=kfold)
    print('The cross_validation score for cat is:',-score.mean())
    et_cat.fit(train_cat,label_cat)
    predictions_cat=et_cat.predict_proba(test_cat)

    columns=et_dog.classes_
    output_dog=pd.DataFrame(predictions_dog,columns=columns)
    output_dog=pd.concat([id_dog,output_dog],axis=1)

    output_cat=pd.DataFrame(predictions_cat,columns=columns)
    output_cat=pd.concat([id_cat,output_cat],axis=1)

    output_combination=pd.concat([output_cat,output_dog],axis=0)
    output_combination_extra=output_combination.sort_values(by='ID',ascending=True)
    print(output_combination_extra)
    output_combination_extra.to_csv('output_comb_extra.csv',index=False)


