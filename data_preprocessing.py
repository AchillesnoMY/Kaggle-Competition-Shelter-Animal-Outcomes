import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from pandas.tseries.holiday import Holiday, HolidayCalendarFactory, USFederalHolidayCalendar, FR

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

# return 0 if the date is a weekend or holiday
# holidays is an index of holiday dates
def is_workday(date, holidays):
    if date in holidays or date.weekday() > 4:
        return 0
    else:
        return 1

#convert age to values in years
def calc_age_in_years(x):
    x = str(x)
    if x == 'nan': return 0
    age = int(x.split()[0])
    if x.find('year') > -1: return age
    if x.find('month')> -1: return age / 12.
    if x.find('week')> -1: return age / 52.
    if x.find('day')> -1: return age / 365.
    else: return 0

#label if an animal is mixed type or not
def get_mix(x):
    x = str(x)
    if x.find('Mix') >= 0: return 'mix'
    return 'not'

#label if an animal is male or female
def get_sex(x):
    x = str(x)
    if x.find('Male') >= 0: return 'male'
    if x.find('Female') >= 0: return 'female'
    return 'unknown'

#label if an animal is neutered or not
def get_neutered(x):
    x = str(x)
    if x.find('Spayed') >= 0: return 'neutered'
    if x.find('Neutered') >= 0: return 'neutered'
    if x.find('Intact') >= 0: return 'intact'
    return 'unknown'

#label encoding
def labelEncoding(data):
    #label encode the rest nominal variables
    for i in data.columns:
        if data[i].dtype=='object' and i!='OutcomeType':

            le=preprocessing.LabelEncoder()
            le.fit(data[i])
            data[i]=le.transform(data[i])
    return data

#for cat, label if a cat is short hair or long hair
def hair_group(breed):
    if breed.find("Shorthair") != -1:
        return 0
    elif breed.find("Longhair") != -1:
        return 1
    else:
        return 2

#split the variable 'Breed'
def breed_group(breed_input):
    breed = str(breed_input)
    if (' ' in breed)==False:
        br =  breed #only 1 word
    else:
        breed_list = breed.split()
        try:
            br = breed_list[2] #fetch last word, for 1 words breed
        except:
            br = breed_list[1] #fetch last word, for 2 words breed
    if br == "Mix":
        return 0
    else:
        return 1

#split the variable 'Color' and return the first string (main color)
def color_group(color):

    try:
        color_type = color.split()
    except:
        return "unknown"
    return str(color_type[0])

#preprocess dog and cat datasets separately including handling missing values, feature engineering etc.
def data_preprocessing(train,test,animalType):

    train_df=train.copy()
    test_df=test.copy()

    '''''''''Handle Outliers'''''''''''

    #one missing value in 'SexuponOutcome' in train data
    train_df.loc[train_df['SexuponOutcome'].isnull(),'SexuponOutcome']='Neutered Male'

    #replace missing values in 'OutcomeSubtype' as 'others'
    train_df=train_df.drop(['OutcomeSubtype'],axis=1)

    # Repalce missing values in 'AgeuponOutcome' in test and train dataset by 1-year
    test_df.loc[test_df['AgeuponOutcome'].isnull(),'AgeuponOutcome']='1 year'
    train_df.loc[train_df['AgeuponOutcome'].isnull(), 'AgeuponOutcome'] = '1 year'

    '''''''''feature engineering'''''''''''

    #remove the letter 'A' in 'AnimalID' variable
    train_df['AnimalID']=train_df['AnimalID'].apply(lambda x: int(x[1:]))

    #create a variable to show whether the animal has a name or not
    train_df['hasName']=train_df['Name'].apply(lambda x:'Yes' if str(x) !='nan' else 'No')
    test_df['hasName'] = test_df['Name'].apply(lambda x: 'Yes' if str(x) !='nan' else 'No')


    train_df=train_df.drop(['Name'],axis=1)
    test_df=test_df.drop(['Name'],axis=1)

    #split 'AgeuponOutcome' as integer time and unit of time without the last letter 's'
    train_df['timeValue']=[int(x.split(' ')[0]) for x in train_df['AgeuponOutcome'].values]
    train_df['UnitOfTime']=[x.split(' ')[1] for x in train_df['AgeuponOutcome'].values]
    train_df['UnitOfTime']=train_df['UnitOfTime'].apply(lambda x: x[0:len(x)-1] if x[-1]=='s' else x)

    test_df['timeValue']=[int(x.split(' ')[0]) for x in test_df['AgeuponOutcome'].values]
    test_df['UnitOfTime']=[x.split(' ')[1] for x in test_df['AgeuponOutcome'].values]
    test_df['UnitOfTime']=test_df['UnitOfTime'].apply(lambda x: x[0:len(x)-1] if x[-1]=='s' else x)

    #calculate the 'timevalue' in days
    multiplier_train=np.zeros(train_df.shape[0])
    multiplier_test=np.zeros(test_df.shape[0])
    for i in range(len(multiplier_train)):
        if train_df['UnitOfTime'].values[i]=='day':
            multiplier_train[i]=1
        elif train_df['UnitOfTime'].values[i]=='week':
            multiplier_train[i]=7
        elif train_df['UnitOfTime'].values[i]=='month':
            multiplier_train[i]=30
        else:
            multiplier_train[i]=365
    for j in range(len(multiplier_test)):
        if test_df['UnitOfTime'].values[j]=='day':
            multiplier_test[j]=1
        elif test_df['UnitOfTime'].values[j]=='week':
            multiplier_test[j]=7
        elif test_df['UnitOfTime'].values[j]=='month':
            multiplier_test[j]=30
        else:
            multiplier_test[j]=365
    train_df['ageInDays']=train_df['timeValue']*multiplier_train
    test_df['ageInDays']=test_df['timeValue']*multiplier_test

    #create new variable 'ageInYears'
    train_df['AgeInYears'] = train_df.AgeuponOutcome.apply(calc_age_in_years)
    test_df['AgeInYears'] = test_df.AgeuponOutcome.apply(calc_age_in_years)

    #drop the variable 'AgeuponOutcome'
    train_df=train_df.drop(['AgeuponOutcome'],axis=1)
    test_df=test_df.drop(['AgeuponOutcome'],axis=1)

    #add a hair group (for cat)
    if animalType=='Cat':

       train_df['hairGroup']=train_df['Breed'].apply(hair_group)
       test_df['hairGroup']=test_df['Breed'].apply(hair_group)
    if animalType=='Dog':

        train_df['aggresiveness']=train_df['Breed'].apply(aggressive)
        test_df['aggresiveness']=test_df['Breed'].apply(aggressive)

        train_df['allergic']=train_df['Breed'].apply(allergic)
        test_df['allergic']=test_df['Breed'].apply(allergic)

        train_df['weight']=train_df['Breed'].apply(weight)
        test_df['weight']=test_df['Breed'].apply(weight)

    #Deal with 'Breed'
    combine_df=pd.concat([train_df,test_df],axis=0)
    breed_type=combine_df.groupby('Breed').size().reset_index()
    breed_type.columns=['Breed','Count']
    breed_type=breed_type.sort_values(by='Count',ascending=False)
    #only keep the breed with frequency greater than 130, the rest are labeled as 'others'
    breed_names=breed_type.loc[breed_type.Count>130,'Breed'].values
    train_df['Breed']=train_df['Breed'].apply(lambda x:'others' if x not in breed_names else x)
    test_df['Breed']=test_df['Breed'].apply(lambda  x:'others' if x not in breed_names else x)

    #Deal with 'color'

    color=combine_df['Color'].value_counts().reset_index()
    color.columns=['Color','Count']
    color=color.sort_values(by='Count',ascending=False)

    #only keep the color with frequency greater than 300, the rest are labeled as 'others'
    color_names=color.loc[color.Count>300,'Color'].values
    train_df['Color']=train_df['Color'].apply(lambda x: 'others' if x not in color_names else x)
    test_df['Color']=test_df['Color'].apply(lambda x: 'others' if x not in color_names else x)

    #create 'ageStatus' feature to show 'old'(>10years), 'middle age'(3years to 10years)  and 'young' (<3 years)
    train_df['ageStatus']='none'
    test_df['ageStatus']='none'
    for i in range(train_df.shape[0]):
        if train_df['ageInDays'].values[i]<1095:
            train_df['ageStatus'].values[i]='young'
        elif train_df['ageInDays'].values[i]>3650:
            train_df['ageStatus'].values[i]='old'
        else:
            train_df['ageStatus'].values[i]='middle age'

    for j in range(test_df.shape[0]):
        if test_df['ageInDays'].values[j]<1095:
            test_df['ageStatus'].values[j]='young'
        elif test_df['ageInDays'].values[j]>3650:
            test_df['ageStatus'].values[j]='old'
        else:
            test_df['ageStatus'].values[j]='middle age'

    #create agesInHours since hour is an important feature
    train_df['ageInHours']=train_df['ageInDays']*24
    test_df['ageInHours']=test_df['ageInDays']*24

    #create new variables 'neutered' and 'sex'
    train_df['sex']=train_df['SexuponOutcome'].apply(get_sex)
    test_df['sex']=test_df['SexuponOutcome'].apply(get_sex)

    train_df['neutered']=train_df['SexuponOutcome'].apply(get_neutered)
    test_df['neutered'] = test_df['SexuponOutcome'].apply(get_neutered)

    #create new variable 'mixOrNot'
    train_df['mixOrNot']=train_df['Breed'].apply(get_mix)
    test_df['mixOrNot']=test_df['Breed'].apply(get_mix)

    train_df=train_df.drop(['SexuponOutcome'],axis=1)
    test_df=test_df.drop(['SexuponOutcome'],axis=1)

    return train_df, test_df

#aggressiveness based on breed type. Most dangerous breeds:
	# Pitbull (55-65 lbs), Rottweiler (100-130 lbs), Husky-type (66 lbs), German
	# Shepherd (100 lbs) , Alaskan Malamute (100 lbs), Doberman pinscher (65-90lbs),
	# chow chow (70 lbs), Great Danes (200pounds), Boxer (70 lbs), Akita (45 kg)
def aggressive(breed):
        if breed.find("Pit Bull") != -1:
            return 1
        elif breed.find("Rottweiler") != -1:
            return 2#1
        elif breed.find("Husky") != -1:
            return 3#1
        elif breed.find("Shepherd") != -1:
            return 4#1
        elif breed.find("Malamute") != -1:
            return 5#1
        elif breed.find("Doberman") != -1:
            return 6#1
        elif breed.find("Chow") != -1:
            return 7#1
        elif breed.find("Dane") != -1:
            return 8#1
        elif breed.find("Boxer") != -1:
            return 9#1
        elif breed.find("Akita") != -1:
            return 10#1
        else:
            return 11#2

#Most allergic breeds:
	#Akita, Alaskan Malamute, American Eskimo, Corgi, Chow-chow, German
	#Shepherd, Great Pyrenees, Labrador, Retriever, Husky
def allergic(breed):
        if breed.find("Akita") != -1:
             return 1
        elif breed.find("Malamute") != -1:
            return 2#1
        elif breed.find("Eskimo") != -1:
            return 3#1
        elif breed.find("Corgi") != -1:
            return 4#1
        elif breed.find("Chow") != -1:
            return 5#1
        elif breed.find("Shepherd") != -1:
            return 6#1
        elif breed.find("Pyrenees") != -1:
            return 7#1
        elif breed.find("Labrador") != -1:
            return 8#1
        elif breed.find("Retriever") != -1:
            return 9#1
        elif breed.find("Husky") != -1:
            return 10#1
        else:
            return 11#2

#weight based on breed type. Most dangerous breeds:
	# Below 100 lbs: Pitbull (55-65 lbs), Husky-type (66 lbs), Doberman pinscher (65-90lbs), Boxer (70 lbs), Akita (45 kg), chow chow (70 lbs)
	# Above 100 lbs: Rottweiler (100-130 lbs), German Shepherd (100 lbs), Alaskan Malamute (100 lbs), Great Danes (200pounds),
def weight(breed):
        if breed.find("Pit Bull") != -1:
            return 1
        elif breed.find("Husky") != -1:
            return 1
        elif breed.find("Doberman") != -1:
            return 1
        elif breed.find("Boxer") != -1:
            return 1
        elif breed.find("Akita") != -1:
            return 1
        elif breed.find("Chow") != -1:
            return 1
        elif breed.find("Rottweiler") != -1:
            return 2
        elif breed.find("Shepherd") != -1:
            return 2
        elif breed.find("Malamute") != -1:
            return 2
        elif breed.find("Dane") != -1:
            return 2
        else:
            return 3

#deal with 'DateTime' variable
def convertDate(train,test):
    #convert the 'Dates'
    train_df=train.copy()
    test_df=test.copy()
    train_df['year']=pd.to_datetime(train_df['DateTime']).dt.year
    test_df['year']=pd.to_datetime(test_df['DateTime']).dt.year

    train_df['month']=pd.to_datetime(train_df['DateTime']).dt.month
    test_df['month']=pd.to_datetime(test_df['DateTime']).dt.month

    train_df['day']=pd.to_datetime(train_df['DateTime']).dt.day
    test_df['day']=pd.to_datetime(test_df['DateTime']).dt.day

    train_df['hour']=pd.to_datetime(train_df['DateTime']).dt.hour
    test_df['hour'] = pd.to_datetime(test_df['DateTime']).dt.hour

    train_df['minute'] = pd.to_datetime(train_df['DateTime']).dt.minute
    test_df['minute'] = pd.to_datetime(test_df['DateTime']).dt.minute

    train_df['quarter']=pd.to_datetime(train_df['DateTime']).dt.quarter
    test_df['quarter']=pd.to_datetime(test_df['DateTime']).dt.quarter

    train_df['week_year']=pd.to_datetime(train_df['DateTime']).dt.weekofyear
    test_df['week_year'] = pd.to_datetime(test_df['DateTime']).dt.weekofyear

    train_df['dayOfWeek']=pd.to_datetime(train_df['DateTime']).dt.dayofweek
    test_df['dayOfWeek']=pd.to_datetime(test_df['DateTime']).dt.dayofweek

    train_df['am_pm']=0
    test_df['am_pm']=0
    for i in range(0, train_df.shape[0]):
        if 8 <= train_df.loc[i,'hour'] <= 12:
            train_df.loc[i,'am_pm'] = 1
        if 12 <= train_df.loc[i,'hour'] <= 14:
            train_df.loc[i,'am_pm'] = 2
        if 14 <= train_df.loc[i,'hour']<= 18:
            train_df.loc[i,'am_pm'] = 3
        if 18 < train_df.loc[i,'hour']:
            train_df.loc[i,'am_pm'] = 4

    for i in range(0, test_df.shape[0]):
        if 8 <= test_df.loc[i,'hour'] <= 12:
            test_df.loc[i,'am_pm'] = 1
        if 12 <= test_df.loc[i,'hour'] <= 14:
            test_df.loc[i,'am_pm'] = 2
        if 14 <= test_df.loc[i,'hour'] <= 18:
            test_df.loc[i,'am_pm']= 3
        if 18 < test_df.loc[i,'hour']:
            test_df.loc[i,'am_pm'] = 4

    train_df['Season'] = 0
    test_df['Season'] = 0

    for i in range(0, test_df.shape[0]):
        if 4 <= test_df.loc[i,'month'] <= 6:
            test_df.loc[i,'Season'] = 1
        if 7 <= test_df.loc[i,'month'] <= 9:
            test_df.loc[i,'Season'] = 2
        if 10 <= test_df.loc[i,'month'] <= 12:
            test_df.loc[i,'Season'] = 3

    for i in range(0, train_df.shape[0]):
        if 4 <= train_df.loc[i,'month']  <= 6:
            train_df.loc[i,'Season']= 1
        if 7 <= train_df.loc[i,'month']  <= 9:
            train_df.loc[i,'Season'] = 2
        if 10 <= train_df.loc[i,'month']  <= 12:
            train_df.loc[i,'Season'] = 3

    train_df['Holidays'] = 0
    test_df['Holidays'] = 0

    for i in range(0, train_df.shape[0]):
        # confederate Heroes
        if train_df.loc[i,'month'] == 1 and train_df.loc[i,'day'] == 19:
            train_df.loc[i,'Holidays']= 1
        # Texas Independance
        if train_df.loc[i,'month'] == 3 and train_df.loc[i,'day'] == 2:
            train_df.loc[i,'Holidays'] = 2
        # San Jacinto Day
        if train_df.loc[i,'month'] == 4 and train_df.loc[i,'day'] == 21:
            train_df.loc[i,'Holidays'] = 3
            # Mothers Day
        if train_df.loc[i,'month'] == 5 and train_df.loc[i,'day']== 10:
            train_df.loc[i,'Holidays'] = 4
            # Emancipation Day
        if train_df.loc[i,'month'] == 6 and train_df.loc[i,'day']== 19:
            train_df.loc[i,'Holidays'] = 5
        # Fathers Day
        if train_df.loc[i,'month'] == 6 and 18 <= train_df.loc[i,'day'] <= 22:
            train_df.loc[i,'Holidays']= 6
        # Lyndon day
        if train_df.loc[i,'month'] == 8 and train_df.loc[i,'day'] == 27:
            train_df.loc[i,'Holidays']= 7
            # Veterans day
        if train_df.loc[i,'month'] == 11 and train_df.loc[i,'day']== 11:
            train_df.loc[i,'Holidays'] = 8
            # ThanksGiving
        if train_df.loc[i,'month'] == 11 and 25 <= train_df.loc[i,'day'] <= 27:
            train_df.loc[i,'Holidays'] = 9
        # Christmas
        if train_df.loc[i,'month'] == 12 and 23 <= train_df.loc[i,'day'] <= 27:
            train_df.loc[i,'Holidays']= 10

    for i in range(0, test_df.shape[0]):
        # confederate Heroes
        if test_df.loc[i,'month']== 1 and test_df.loc[i,'day'] == 19:
            test_df.loc[i,'Holidays'] = 1
        # Texas Independance
        if test_df.loc[i,'month'] == 3 and test_df.loc[i,'day'] == 2:
            test_df.loc[i,'Holidays']= 2
        # San Jacinto Day
        if test_df.loc[i,'month'] == 4 and test_df.loc[i,'day']== 21:
            test_df.loc[i,'Holidays'] = 3
            # Mothers Day
        if test_df.loc[i,'month'] == 5 and test_df.loc[i,'day']== 10:
            test_df.loc[i,'Holidays'] = 4
            # Emancipation Day
        if test_df.loc[i,'month'] == 6 and test_df.loc[i,'day']== 19:
            test_df.loc[i,'Holidays'] = 5
        # Fathers Day
        if test_df.loc[i,'month'] == 6 and 18 <= test_df.loc[i,'day'] <= 22:
            test_df.loc[i,'Holidays'] = 6
        # Lyndon day
        if test_df.loc[i,'month'] == 8 and test_df.loc[i,'day']== 27:
            test_df.loc[i,'Holidays'] = 7
            # Veterans day
        if test_df.loc[i,'month']== 11 and test_df.loc[i,'day']== 11:
            test_df.loc[i,'Holidays']= 8
            # ThanksGiving
        if test_df.loc[i,'month'] == 11 and 25 <= test_df.loc[i,'day'] <= 27:
            test_df.loc[i,'Holidays']= 9
        # Christmas
        if test_df.loc[i,'month'] == 12 and 23 <= test_df.loc[i,'day'] <= 27:
            test_df.loc[i,'Holidays']= 10

    #drop 'Dates'
    train_df.drop(['DateTime'],axis=1,inplace=True)
    test_df.drop(['DateTime'],axis=1,inplace=True)
    return train_df,test_df

#split datasets into dog and cat datasets. Do data preprocessing and combine them afterwards.
def cat_dog_datasets(train,test):

    train_df,test_df=convertDate(train,test)
    train_dog=train_df.loc[train_df['AnimalType']=='Dog',:]
    test_dog=test_df.loc[test_df['AnimalType']=='Dog',:]
    train_cat=train_df.loc[train_df['AnimalType']=='Cat',:]
    test_cat=test_df.loc[test_df['AnimalType']=='Cat',:]

    train_dog,test_dog=data_preprocessing(train_dog,test_dog,'Dog')
    train_cat,test_cat=data_preprocessing(train_cat,test_cat,'Cat')

    train_dog=labelEncoding(train_dog)
    test_dog=labelEncoding(test_dog)

    train_cat=labelEncoding(train_cat)
    test_cat=labelEncoding(test_cat)

    train_dog.to_csv('train_dog.csv',index=False)
    test_dog.to_csv('test_dog.csv',index=False)
    train_cat.to_csv('train_cat.csv',index=False)
    test_cat.to_csv('test_cat.csv',index=False)





