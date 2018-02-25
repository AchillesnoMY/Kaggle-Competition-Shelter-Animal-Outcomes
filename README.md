# Kaggle Competition: Shelter-Animal-Outcomes

Solution to kaggle Shelter-Animal-Outcomes competition.<br> 
Final Rank: 123th<br>
Leaderboard Score: 0.71425

## Overview:

---------------------

[Austin Animal Center](http://www.austintexas.gov/department/aac) share their animal information including color, breed etc. and wish kaggers could focus on the dataset provided to predict the probabilities of a set of outcomes for each animal so that shelters could provide extra care to those animals who need to it more. The details of this competition and datasets can be found at https://www.kaggle.com/c/shelter-animal-outcomes


## Method Descriptions:

-------------------------

The first model I used is Random Forest with 1000 number of trees and a maximum depth of 8 to give me a benchmark score.
I spent a lot of time on construction of new features such as 'ageInYears', 'ageInHours','neutered' etc, and from the suggestions provided by [vzaretsk](https://github.com/vzaretsk/kaggle-animal-shelter), I also included some external features, specifically dog breed size, energy, and popularity, available on the American Kennel Club website (http://www.akc.org/). However, these new added features from American Kennel Club make trivial contribution to my Random Forest. 

I removed some nonsignificant features with feature importance f-score (from Random Forest) less than 0.02, and this approach largely increases my leaderboard score. 

Instead of predicting the dog and cat datasets as a whole, I predict them seperately by using different Random Forests models with different parameters and then combine their outputs together.

Besides Random Forest, I also tried Logistical Regression, ExtraTree Classifier, gradientBoosting Classifier. The performance of gradientBoosting Classifier is the best among all. 

At last, I used a simple ensemble by assigning two different weights (0.7,0.3) to outputs from Random Forest and gradientBoosting Classifer and it gave me the final score of 0.71425.
