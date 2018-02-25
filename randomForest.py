import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV


#Hyperparameters tuning for both dog and cat datasets
def rf_parameterTuning():
    train_cat = pd.read_csv('train_cat.csv')
    train_cat = train_cat.drop(['AnimalID'], axis=1)
    label_cat = train_cat['OutcomeType']
    train_cat = train_cat.drop(['OutcomeType'], axis=1)

    train_dog = pd.read_csv('train_dog.csv')
    train_dog = train_dog.drop(['AnimalID'], axis=1)
    label_dog=train_dog['OutcomeType']
    train_dog=train_dog.drop(['OutcomeType'],axis=1)

    #tuning the parameters for dog dataset first
    seed=123
    estimators_chosen=[20,100,500,800,1000]
    min_samples_leaf=[2,4,6]
    max_depth=[8,10,12]
    rf=RandomForestClassifier(max_features='sqrt',max_depth=8)
    #tuning the estimators number and learning rate for dog
    for i in estimators_chosen:
        for j in min_samples_leaf:
            for k in max_depth:
               rf.set_params(n_estimators=i,min_samples_leaf=j,max_depth=k)
               kfold=KFold(n_splits=10,random_state=seed)
               score=cross_val_score(rf,X=train_dog,y=label_dog,scoring='neg_log_loss',cv=kfold)
               score=(-score.mean())
               print('For dog dataset:')
               print('The n_estimators=%d, the min_samples_leaf=%.3f, the max_depth=%d, give the score=%f'%(i,j,k,score))
    #The n_estimators=1000, the min_samples_leaf=2.000, the max_depth=12, give the score=0.926948
    #tuning the parameters for cat dataset
    seed = 123
    estimators_chosen = [20, 100, 500, 800, 1000]
    min_samples_leaf = [2, 4, 6]
    max_depth = [8, 10, 12]
    rf = RandomForestClassifier(max_features='sqrt', max_depth=8)
    # tuning the estimators number and learning rate for dog
    for i in estimators_chosen:
        for j in min_samples_leaf:
            for k in max_depth:
                rf.set_params(n_estimators=i, min_samples_leaf=j, max_depth=k)
                kfold = KFold(n_splits=10, random_state=seed)
                score = cross_val_score(rf, X=train_cat, y=label_cat, scoring='neg_log_loss', cv=kfold)
                score = (-score.mean())
                print('For cat dataset:')
                print('The n_estimators=%d, the min_samples_leaf=%.3f, the max_depth=%d, give the score=%f' % (
                i, j, k, score))
    #The n_estimators=1000, the min_samples_leaf=2.000, the max_depth=12, give the score=0.521921


#plot feature importance
def plotFeatureImportance(rf,train):
    feature_importance=pd.Series(rf.feature_importances_,index=train.columns.values)
    feature_importance.sort_values(ascending=False,inplace=True)
    sns.set(style='darkgrid')
    feature_importance.plot(kind='bar')
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig('feature_importance_rf.png')
    plt.show()

#remove nonsignificant features with f-score less than 0.02
def removeUnimporantFeat(train,test,rf):
    feature_importance=pd.Series(rf.feature_importances_,index=train.columns.values)
    feature_importance.sort_values(ascending=False,inplace=True)
    #print(feature_importance)
    removed_features=feature_importance.index.values[feature_importance.values<0.02]
    train=train.drop(removed_features,axis=1)
    test=test.drop(removed_features,axis=1)
    return train,test

seed=123

#istotonic calibaration
def iso_calibration(rf,train,label,test):

    # check calibration's performance
    rf_predictions = pd.read_csv('output_rf_test.csv')
    rf_predictions = rf_predictions.drop('ID', axis=1)
    rf_predictions = rf_predictions.values
    #rf_predictions=rf_predictions.flatten()
    #plot the figure before calibration
    sns.set(style='darkgrid')
    plt.hist(rf_predictions, range=(0, 1), bins=10, lw=2)
    plt.xlabel('Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Before Calibration')
    plt.savefig('no_calibration.png')
    plt.show()
    # isotonic calibration
    calibrated_rf_iso = CalibratedClassifierCV(rf, method='isotonic', cv=10)
    calibrated_rf_iso.fit(train, label)
    pred_Calibration_iso = calibrated_rf_iso.predict_proba(test)

    #local cross validation and plot the figure after calibration
    kfold = KFold(n_splits=10, random_state=seed)
    score_iso = cross_val_score(calibrated_rf_iso, train, label, scoring='neg_log_loss', cv=kfold)
    print('Log loss for isotonic calibration:', -score_iso.mean())

    sns.set(style='darkgrid')
    plt.hist(pred_Calibration_iso,range=(0,1),bins=10,lw=2)
    plt.xlabel('Probability',fontsize=12)
    plt.ylabel('Frequency',fontsize=12)
    plt.title('After Calibration')
    plt.savefig('calibration.png')
    plt.show()

    return pred_Calibration_iso

#predict dog and cat datasets separately and combine them together
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

    rf_dog=RandomForestClassifier(n_estimators=1000,max_features='sqrt',max_depth=12,min_samples_leaf=2,random_state=
                              seed)
    rf_cat=RandomForestClassifier(n_estimators=1000,max_features='sqrt',max_depth=12,min_samples_leaf=2,random_state=
                              seed)

    rf_dog.fit(train_dog,label_dog)
    plotFeatureImportance(rf_dog,train_dog)
    train_dog,test_dog=removeUnimporantFeat(train_dog,test_dog,rf_dog)
    #cross-validation for dog
    kfold=KFold(n_splits=10,random_state=seed)
    score=cross_val_score(rf_dog,train_dog,label_dog,scoring='neg_log_loss',cv=kfold)
    print('The cross_validation score for dog is:',-score.mean())#0.920256265035138 0.9201144465973398 0.9199378904858027
    #rf_dog.fit(train_dog,label_dog)
    #predictions_dog=rf_dog.predict_proba(test_dog)
    predictions_dog=iso_calibration(rf_dog,train_dog,label_dog,test_dog)

    rf_cat.fit(train_cat,label_cat)
    plotFeatureImportance(rf_cat,train_cat)
    train_cat,test_cat=removeUnimporantFeat(train_cat,test_cat,rf_cat)
    #cross-validation for cat
    kfold=KFold(n_splits=10,random_state=seed)
    score=cross_val_score(rf_cat,train_cat,label_cat,scoring='neg_log_loss',cv=kfold)
    print('The cross_validation score for cat is:',-score.mean())#0.5218890016130225 0.5191591317713868 0.5114550970472586
    #rf_cat.fit(train_cat,label_cat)
    #predictions_cat=rf_cat.predict_proba(test_cat)
    predictions_cat = iso_calibration(rf_cat, train_cat, label_cat, test_cat)

    columns=rf_dog.classes_
    output_dog=pd.DataFrame(predictions_dog,columns=columns)
    output_dog=pd.concat([id_dog,output_dog],axis=1)

    output_cat=pd.DataFrame(predictions_cat,columns=columns)
    output_cat=pd.concat([id_cat,output_cat],axis=1)

    output_combination=pd.concat([output_cat,output_dog],axis=0)

    output_combination_rf=output_combination.sort_values(by='ID',ascending=True)
    print(output_combination_rf)
    output_combination_rf.to_csv('output_comb_rf2.csv',index=False)

#predict cat and dog datasets as a whole
def predictSingle():
    train = pd.read_csv('newTrain.csv')
    train = train.drop(['AnimalID'], axis=1)
    label = train['OutcomeType']
    train = train.drop(['OutcomeType'], axis=1)
    test = pd.read_csv('newTest.csv')
    id = test.ID
    test = test.drop(['ID'], axis=1)

    rf=RandomForestClassifier(n_estimators=800,max_features='sqrt',max_depth=8,min_samples_leaf=2,random_state=
                          seed)
    rf.fit(train,label)
    plotFeatureImportance(rf,train)
    train,test=removeUnimporantFeat(train,test,rf)
    rf.fit(train,label)
    #cross-validation for rf
    kfold=KFold(n_splits=10,random_state=seed)
    score=cross_val_score(rf,train,label,scoring='neg_log_loss',cv=kfold)
    print(-score.mean()) #0.797104243338149 0.7967723129702915
    columns=rf.classes_
    predictions=rf.predict_proba(test)
    output_rf=pd.DataFrame(predictions,columns=columns)
    output_rf=pd.concat([id,output_rf],axis=1)
    output_rf.to_csv('output_rf.csv',index=False)




