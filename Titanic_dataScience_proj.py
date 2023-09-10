import numpay as np             #Import Numpay Library for numerical computations  In-1
import pandas as pd             #Import pandas Library for data manipulation and analysis
import seaborn as sns           #Import Seaborn Library for stastical data visualisation
import matplotlib.pyplot as plt  #Import Matplotlib Library for data visualization
from sklearn.model_selection import train_test_split   #Import train_test_split function for splitting data
from sklearn.metrics import confusion_matrix          #Import confusion_matrix function for detailed classification metrics
from sklearn.metrics import classification_report      #Import classification_rrepot function for detailed clssification metrics
from sklearn.linear_model import LogisticRegression    #Import LogisticRegression class from Logistics regression
a = pd.read_csv("Titanic-Dataset.Csv") # In-2
df =pd.DataFrame(a) #In-3
df.head(5)
#shape and size of data present In-5
df.shape  
#features of data In-6  
df.columns 
#Stastical summery data In-7
df.describe(include='all')     
missing_values=df.isnull().sum()
missing_values.sort_values(ascending=false) #In-8
def count_plot(feature):  
#the count_plot Function creats a barplot showing  countof unique values of categorical variable    
sns.countplot(x=feature ,data = df)
plt.show() #In-9
columns=['Survived','pclass','Sex','SibSp','Embarked'] # Columns to be visualized
for i in columns:
count_plot(i) #In-10
df.head() #In-11
df.drop(['passengerId','Name','Cabin','Ticket',],axis=1,inplace=True)    #In-12
df['Age'].fillna(df['Age'].mean(),inplace=True)
df.isnull().sum() #filling missing values In-13
df #In-14
#Creating dummy variables for the 'Sex' column using get_dummies()Functions to transform categorical variables into numerical
Sex =pd.get_dummies(df['Sex'],drop_first=True) 
Sex.head() #In-15
#Creating dummy variables for the 'Embarked' column
embark=pd.get_dummies(df['Embarked'],drop_first=True)
embark.head()  # In-16
#Creating dummy variables for the 'pclass' column
pclass=pd.get_dummies(df['pclass'],drop_first=True)
pclass.head() #In-17
df.head() #In-18
df.drop(["Sex","Embarked","pclass"],axis=1,inplace=True)
df.head() #In-19
#Concatenating the original df with the dummy variables
df=pd.concat([df,sex,embark,pclass],axis=1)
#Concatenating the features to String datatype
df.columns=df.columns.astype(str)
df.head()    #in-20
#Creating the feature variable x by dropping the 'Survived'column from df
X=df.drop(['Survived'],axis=1)
#Creating target variable y by assigning the 'Survived' column
y=df['Survived']      #In-21
X_train,X_test,y_train,y_test = train_test_split( x , y ,test_size=0.3,random_state = 42) #In-22
#Creating and training the logistic regression model #In-23
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
model.score(X_train,y_train)   #In-24
model.score(X_test,y_test)       # In-25
#using the model to predict the labels
y_predicted=model.predict(x_test)
#Evalute the performance of a model by calculating the confusion  matrix   In-26
confusion_matrix(y_test, y_predicted)
print(classification_report(y_test,y_predicted))