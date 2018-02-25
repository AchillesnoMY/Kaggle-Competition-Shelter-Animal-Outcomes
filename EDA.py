import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#import train, test datasets.
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#count missing values for both datasets
print('Missing values counts for training dataset:')
print(train.isnull().sum(axis=0))
print('Missing values counts for testing dataset:')
print(test.isnull().sum(axis=0))

#count number of dogs and cats
def countAnimalTypes(train):
    plt.figure(figsize=(6,4))
    sns.set(style='darkgrid')
    sns.countplot(train['AnimalType'],palette='Set3')
    plt.savefig('animial_count.png')
    plt.show()

#count the outcomes
def countOutcomeType(train):
    plt.figure(figsize=(6,4))
    sns.set(style='darkgrid')
    sns.countplot(train['OutcomeType'],palette='Set3')
    plt.savefig('outcome_type.png')
    plt.show()

#plot the distributions of animal type and outcomes
def outcomeTypeAnalysis(train):
    sns.set(style='darkgrid')
    f,(ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
    plt.tight_layout()
    sns.countplot(data=train, x='OutcomeType', hue='AnimalType', ax=ax1)
    sns.countplot(data=train, x='AnimalType', hue='OutcomeType', ax=ax2)
    plt.savefig('outcome_analysis.png')
    plt.show()


#plot missing frequency
def plotMissing(train,test):
    train_miss=train.isnull().sum(axis=0).reset_index()
    train_miss.columns=['feature','missing']
    test_miss=test.isnull().sum(axis=0).reset_index()
    test_miss.columns=['feature','missing']
    #plot the training set
    sns.set(style='darkgrid')
    plt.subplot(1,2,1)
    sns.barplot(x=train_miss.feature,y=train_miss.missing,palette='Set2')
    plt.xticks(rotation='vertical')
    plt.title('Train')
    plt.tight_layout()
    #plot the testing set
    plt.subplot(1,2,2)
    sns.barplot(x=test_miss.feature,y=test_miss.missing,palette='Set2')
    plt.xticks(rotation='vertical')
    plt.title('Test')
    plt.tight_layout()
    plt.savefig('missing_count.png')
    plt.show()

#count the animals of different sex.
def countSexUponOutcome(train):
    plt.figure(figsize=(6,4))
    sns.set(style='darkgrid')
    sns.countplot(train['SexuponOutcome'],palette='Set3')
    plt.savefig('count_sexuponOutcome.png')
    plt.show()


#simplify the 'SexuponOutcome' to 'male','female' and 'unknown' only.
def get_sex(x):
    x = str(x)
    if x.find('Male') >= 0: return 'male'
    if x.find('Female') >= 0: return 'female'
    return 'unknown'

#plot the distributions of sex and outcomes
def sexAnalysis(train):
    train['sex'] = train['SexuponOutcome'].apply(get_sex)
    sns.set(style='darkgrid')
    f,(ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
    plt.tight_layout()
    sns.countplot(data=train, x='OutcomeType', hue='sex', ax=ax1)
    sns.countplot(data=train, x='sex', hue='OutcomeType', ax=ax2)
    plt.savefig('sex_analysis.png')
    plt.show()

#simplify the 'sexuponOutcome' to 'neutered','intact' and 'unknown'
def get_neutered(x):
    x = str(x)
    if x.find('Spayed') >= 0: return 'neutered'
    if x.find('Neutered') >= 0: return 'neutered'
    if x.find('Intact') >= 0: return 'intact'
    return 'unknown'

#plot the distributions of neutered and outcomes
def neuteredAnalysis(train):
    train['neutered']=train['SexuponOutcome'].apply(get_neutered)
    sns.set(style='darkgrid')
    f,(ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
    plt.tight_layout()
    sns.countplot(data=train, x='OutcomeType', hue='neutered', ax=ax1)
    sns.countplot(data=train, x='neutered', hue='OutcomeType', ax=ax2)
    plt.savefig('neutered_analysis.png')
    plt.show()



