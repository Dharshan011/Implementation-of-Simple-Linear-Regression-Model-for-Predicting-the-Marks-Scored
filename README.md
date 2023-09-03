# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## Step
Upload the file to your cell.
## Step
Type the required program.
## Step 
Print the program.
## Step
End the program.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DHARSHAN V
RegisterNumber:  212222230031

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Rama E.K. Lekshmi
RegisterNumber: 212222240082
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X = df.iloc[:,:-1].values
X

y = df.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

y_pred

y_test

plt.scatter(X_train,y_train,color="green")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,y_test,color="grey")
plt.plot(X_test,regressor.predict(X_test),color="purple")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
*/
```

## Output:
![Screenshot 2023-09-03 194240](https://github.com/Dharshan011/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497491/d2bf6ec2-eaf6-49f5-a4b3-ffadce381ada)


![Screenshot 2023-09-03 194246](https://github.com/Dharshan011/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497491/7a840b4a-22af-498a-8f45-f1d02322f8ee)



![Screenshot 2023-09-03 194253](https://github.com/Dharshan011/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497491/e82c6591-f848-47ee-aaa7-7e51932c4de4)

![Screenshot 2023-09-03 194305](https://github.com/Dharshan011/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497491/a07672f6-b554-4535-9e4b-f323c8b0b8db)



![Screenshot 2023-09-03 194313](https://github.com/Dharshan011/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497491/0cd1439d-a876-4d02-840e-36a3219d424b)

![Screenshot 2023-09-03 194320](https://github.com/Dharshan011/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497491/48e61d77-56e9-4cb6-b1c5-0d99045ac6f3)






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
