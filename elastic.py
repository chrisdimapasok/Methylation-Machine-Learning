#Epignetic Clock Analysis

import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet


#Load in methlation matrix data
pd.set_option('display.max_columns', 100) #24444 
df= pd.read_csv('MethMatrix40_10-100.csv')
df


#Load in sample characteristics dataset
df1=pd.read_csv('ProsperSalivaTraits34_030520mjt.csv')
df1


#Quantify ages of population
def sample_ages():
    ages= df1['age']
    plt.hist(ages, bins=5, ec='black')
    plt.title('Age of Samples')
    plt.xlabel('Ages')
    plt.ylabel('Number of individuals')
    plt.show()
    mean_ages= df1['age'].mean()
    print('Mean Age:', mean_ages)
sample_ages()


#Quantify weights of population
def sample_weights():
    weights= df1['weight']
    plt.hist(weights, bins=10, ec='black')
    plt.xlabel('Weights')
    plt.ylabel('Number of Individuals')
    plt.title('Weights of Individuals')
    plt.show()
    mean_weight= df1['weight'].mean()
    print('Mean Weight:', mean_weight)
sample_weights()

#Separate by gender
def sample_genders():
    num_ind= df1['female']
    num_ind
    gender= ['Male', 'Female']
    males= []
    females=[]
    for i in num_ind:
        if i== 0: 
            males.append(i)
        else:
            females.append(i)

    males= males.count(0)
    females=females.count(1)

    num= [males,females]
    gender= ['Male', 'Female']
    plt.bar(gender, num, color=('red'))
    plt.title('Gender of Individuals')
    plt.xlabel('Gender')
    plt.ylabel('# of Individuals')
    
sample_genders()    


#Partition percent of smokers vs non smokers in pie chart
def smokers():
    list= df1['smoke_tobacco'].tolist()
    list

    smokers= list.count(1)
    smokers

    nonsmokers= list.count(0)
    nonsmokers

    smkvsnon= ['Smokers', 'Nonsmokers']
    smk= [smokers,nonsmokers]
    explode = (0.1, 0)

    plt.pie(smk, labels= smkvsnon, explode= explode, autopct='%1.1f%%')
    plt.title('Percent of Smokers vs Non-Smokers')
    plt.show()

smokers()


#Process and merge data into one dataframe
dfIdAge
dfIdAge= pd.DataFrame(columns=["CGmapID","age"]) 
dfIdAge

dfIdAge["CGmapID"]= df1["CGmapID"]
dfIdAge["age"]= df1["age"]
dfIdAge

mergedDf= pd.merge(df, dfIdAge, how="inner", left_on= "sampleID",right_on= "CGmapID") 
mergedDf 


mergedDf= mergedDf.drop(["sampleID","CGmapID"], axis=1) 
mergedDf


#Set X and y 
X= mergedDf.loc[:, mergedDf.columns != 'age'] 
y= mergedDf['age']



#Ridge Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2)

def Ridge_Reg():
    from sklearn.linear_model import Ridge
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    Ridge_test_score = ridge.score(X_test, y_test)
    print(Ridge_test_score)

Ridge_Reg()


#Lasso Regression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2)


def Lasso_reg():
    from sklearn.linear_model import Lasso
    lasso= Lasso()
    lasso.fit(X_train, y_train)
    Lasso_test_score= lasso.score(X_test, y_test)
    print(Lasso_test_score)

Lasso_reg()   


#ELASTIC REGRESSION
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2)

def Elastic_reg():
    elastic= ElasticNet()
    elastic.fit(X_train, y_train)
    elastic_test_score= elastic.score(X_test, y_test)
    print(elastic_test_score)

Elastic_reg()


#Crossvalscore mean squared error
from sklearn.model_selection import cross_val_score
print(cross_val_score(ElasticNet(), X,y, scoring= "neg_mean_squared_error", cv=5))
print(cross_val_score(Ridge(), X,y, scoring= "neg_mean_squared_error", cv=5))
print(cross_val_score(Lasso(), X,y, scoring= "neg_mean_squared_error", cv=5))


#Crossvalscore on test sets
print(cross_val_score(elastic, X_test, y_test, cv=2))
print(cross_val_score(Ridge(), X_test, y_test, cv=2))
print(cross_val_score(Lasso(), X_test,y_test, cv=2))


#Define variables for scoring
elastic_score= cross_val_score(ElasticNet(), X_test, y_test, cv=2)
Ridge_score= cross_val_score(Ridge(), X_test, y_test, cv=2)
Lasso_score= cross_val_score(Lasso(), X_test, y_test, cv=2)


#Average scores for each model 
def avg_elastic_score():
    sum_score= sum(elastic_score)
    avg_score= sum_score/len(elastic_score)
    print(avg_score)
    
avg_elastic_score()


def avg_Ridge_score():
    sum_score= sum(Ridge_score)
    avg_score= sum_score/len(Ridge_score)
    print(avg_score)

avg_Ridge_score()


def avg_Lasso_score():
    sum_score= sum(Lasso_score)
    avg_score= sum_score/len(Lasso_score)
    print(avg_score)
    
avg_Lasso_score()


#Another Elastic Net model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2)
elastic= ElasticNet(alpha= .1, l1_ratio=0.5, max_iter= 10000).fit(X_train, y_train)
y_pred = elastic.predict(X_test)
y_pred


#Score model
elastic.score(X_test, y_test)


#Scatter plot
plt.scatter(y_test, y_pred)


#Set y_test and y_pred as arrays
y_test_array= np.array(y_test)
y_pred_array= np.array(y_pred)


#Best fit line for graph
def best_fit_line(y_test, y_pred):
    m, b= np.polyfit(y_test, y_pred, deg=1)
    line = m*y_array + b
    return line
line = best_fit_line(y_test, y_pred)
plt.plot(y_test, y_pred, 'o')
plt.plot(y, line, '--')
plt.title ('Test Set Age vs Predicted Age')
plt.xlabel('Test Set Age')
plt.ylabel('Predicted Age')


#Optimize parameters 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


#lambda magnitude (of the penalty)
def opt_param():
    
    param = {
        "alpha": [0.1, 0.2,0.3,0.4,0.5,0.6, 0.7,0.8,0.9,1],
     "l1_ratio": (0.5, 1.0, 0.0, 0.1)}
    
    param_search= RandomizedSearchCV(elastic, param, cv= 5, iid=False)
    grid_result= param_search.fit(X,y)
    best_params = grid_result.best_params_
    
    print(best_params)

opt_param()


#Generate new model using optimal parameters
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2)
new_elastic= ElasticNet(alpha= 0.4, l1_ratio=0.0).fit(X_train, y_train)
score = new_elastic.score(X_test, y_test)
score


#Set X and y as arrays for leave one out cross validation
X_array= np.array(X)
y_array= np.array(y)


#Leave one out cross validation
from sklearn.model_selection import LeaveOneOut
loocv= LeaveOneOut()

est_y= []
for train, test in loocv.split(X):
    X_train, X_test= X_array[train], X_array[test]
    y_train, y_test= y_array[train], y_array[test]
    new_elastic.fit(X_train,y_train)
    y_pred= elastic.predict(X_test)
    est_y.append(y_pred)


#Plot results
plt.scatter(y, est_y)
plt.plot()


#Best fit line
def best_fit_line(y_array, new_y):
    m, b= np.polyfit(y_array, new_y, deg=1)
    line = m*y_array + b
    return line
line = best_fit_line(y_array, new_y)
plt.plot(y, est_y, 'o')
plt.plot(y, line, '--')
plt.title ('Chronological Age vs Epigenetic Age')
plt.xlabel('Chronological Age')
plt.ylabel('Epigenetic Age')
plt.plot(y, line, '--', dashes= (4,50))


#Get model score
r2_score(y, est_y)




