import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#hyperparameters tuning for logistical regression
def logistic_ParameterTuning(train,label):
    seed=123
    cv_results=[]
    penalty_parameters=[1.0,5.0,10.0,100.0]
    lg=LogisticRegression(penalty='l2',dual=False,C=1.0,solver='saga',multi_class='multinomial')
    for i in penalty_parameters:
        lg.set_params(C=i)
        kfold=KFold(n_splits=10,random_state=seed)
        result=cross_val_score(lg,train,label,scoring='neg_log_loss',cv=kfold)
        cv_results.append(-result.mean())
    print('The cross-validation scores:', cv_results)
    sns.set(style='darkgrid')
    plt.plot(penalty_parameters,cv_results)
    plt.xlabel('estimators',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.show()



#predict cat and dog separately
def predictSeparately():
    # dog and cat datasets
    seed=123
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

    log_dog=LogisticRegression(penalty='l2',dual=False,C=1.0,solver='saga',multi_class='multinomial')
    log_cat=LogisticRegression(penalty='l2',dual=False,C=1.0,solver='saga',multi_class='multinomial')

    log_dog.fit(train_dog,label_dog)
    #cross-validation for dog
    kfold=KFold(n_splits=10,random_state=seed)
    score=cross_val_score(log_dog,train_dog,label_dog,scoring='neg_log_loss',cv=kfold)
    print('The cross_validation score for dog is:',-score.mean())
    predictions_dog=log_dog.predict_proba(test_dog)

    log_cat.fit(train_cat,label_cat)
    #cross-validation for cat
    kfold=KFold(n_splits=10,random_state=seed)
    score=cross_val_score(log_cat,train_cat,label_cat,scoring='neg_log_loss',cv=kfold)
    print('The cross_validation score for cat is:',-score.mean())#0.5515420038579097
    predictions_cat=log_cat.predict_proba(test_cat)

    columns=log_dog.classes_
    output_dog=pd.DataFrame(predictions_dog,columns=columns)
    output_dog=pd.concat([id_dog,output_dog],axis=1)

    output_cat=pd.DataFrame(predictions_cat,columns=columns)
    output_cat=pd.concat([id_cat,output_cat],axis=1)

    output_combination=pd.concat([output_cat,output_dog],axis=0)

    output_combination_log=output_combination.sort_values(by='ID',ascending=True)
    print(output_combination_log)
    output_combination_log.to_csv('output_comb_log.csv',index=False)

