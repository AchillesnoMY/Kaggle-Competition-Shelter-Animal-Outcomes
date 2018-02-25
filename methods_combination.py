import pandas as pd

results_logistic=pd.read_csv('output_comb_log.csv')
results_randomForest=pd.read_csv('output_comb_rf2.csv')
results_extraTree=pd.read_csv('output_comb_extra.csv')
results_gradBoosting=pd.read_csv('output_comb_gbc.csv')
id=results_gradBoosting['ID']
print(results_randomForest)
results_logistic=results_logistic.drop(['ID'],axis=1)
results_randomForest=results_randomForest.drop(['ID'],axis=1)
results_extraTree=results_extraTree.drop(['ID'],axis=1)
results_gradBoosting=results_gradBoosting.drop(['ID'],axis=1)
#results_final=results_gradBoosting*0.6+results_randomForest*0.3+0.1*results_extraTree
results_final=results_gradBoosting*0.7+results_randomForest*0.3
results_final=pd.concat([id,results_final],axis=1)
print(results_final)
results_final.to_csv('results_final.csv',index=False)



