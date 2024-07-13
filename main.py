import pandas as p
import numpy as np
import matplotlib.pyplot  as plt
from sklearn.tree import plot_tree
import seaborn as sn
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor , GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


dt=p.read_csv("data.csv")
print("Shape of the dataset : ",dt.shape)
print(dt.head())
print(dt.describe())


dt["verification.result"]=(dt["verification.result"]==1).astype(int)
print(dt.info())

print("\n\nNo of True entries : ",(dt["verification.result"]==1).sum().sum())
print("No of False entries : ",(dt["verification.result"]==0).sum().sum())


corr=np.corrcoef(dt.values.T)
hm=sn.heatmap(corr,annot=True)
plt.show()

sn.pairplot(dt)
plt.show()
new_column_names = [0, 1, 2, 3, 4, 5, 6, 7, 8]
for i, col in enumerate(dt.columns):
    dt.rename(columns={col: new_column_names[i]}, inplace=True)



tar=dt[7]
inp=dt.drop(columns=[1,2,7,8])

xtrain,xtest,ytrain,ytest=train_test_split(inp,tar,test_size=0.25)



ros=RandomOverSampler()
dt1=dt[7]

print("\n\nTrain input shape :",xtrain.shape)
x,y=ros.fit_resample(xtrain,ytrain)
xtrain=np.hstack((x,np.reshape(y,(-1,1))))
xtrain = p.DataFrame(xtrain)
# print(xtrain[5].value_counts())
ytrain = xtrain[5]  
xtrain.drop(columns=[5],inplace=True)
print("Train input shape after ROS :",xtrain.shape)

print("\n","----"*20,"Applying Classification Algorithms","----"*20,"\n")



print("\n","**"*20,"KNN Classifier","**"*20)
knn=KNeighborsClassifier()
knn.fit(xtrain,ytrain)
knntrainr=knn.predict(xtrain)
knntestr=knn.predict(xtest)
print("#"*15,"Training","#"*15,"\t"*2,"#"*15,"Testing","#"*15)
print("Training accuracy : ",accuracy_score(ytrain,knntrainr),"\t"*2," Testig accuracy   : ",accuracy_score(ytest,knntestr))
print("MSE               : ",mean_squared_error(ytrain,knntrainr),"\t"*2," MSE               : ",mean_squared_error(ytest,knntestr))
print("R2_score          : ",r2_score(ytrain,knntrainr),"\t"*2," R2_score          : ",r2_score(ytest,knntestr))



print("\n\n","**"*20,"Decision Tree Classifier","**"*20)
drc=DecisionTreeClassifier()
drc.fit(xtrain,ytrain)
drc_train=drc.predict(xtrain)
drc_test=drc.predict(xtest)
print("#"*15,"Training","#"*15,"\t"*3," ","#"*15,"Testing","#"*15)
print("Training acc : ",accuracy_score(ytrain,drc_train),"\t"*4,"   Testing acc  : ",accuracy_score(ytest,drc_test))
print("MSE          : ",mean_squared_error(ytrain,drc_train),"\t"*4,"   MSE          : ",mean_squared_error(ytest,drc_test))
print("R2_Score     : ",r2_score(ytrain,drc_train),"\t"*4,"   R2_Score     : ",r2_score(ytest,drc_test))

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(drc, filled=True, feature_names=inp.columns, class_names=["0", "1"])  # Modify class_names as per your data
plt.show()



print("\n\n","**"*20,"Gradient Boosting Classifier","**"*20)
gbc=GradientBoostingClassifier(max_depth=6,max_leaf_nodes=32)
gbc.fit(xtrain,ytrain)
gbc_train=gbc.predict(xtrain)
gbc_test=gbc.predict(xtest)
print("#"*15,"Training","#"*15,"\t"*3,"      ","#"*15,"Testing","#"*15)
print("Training acc : ",accuracy_score(ytrain,gbc_train),"\t"*5,"Testing acc  : ",accuracy_score(ytest,gbc_test))
print("MSE          : ",mean_squared_error(ytrain,gbc_train),"\t"*5,"MSE          : ",mean_squared_error(ytest,gbc_test))
print("R2_Score     : ",r2_score(ytrain,gbc_train),"\t"*5,"R2_score     : ",r2_score(ytest,gbc_test))



print("\n\n","----"*21,"Applying Regression Algorithms","----"*21,"\n")

tar2=dt[8]
inp2=dt.drop(columns=[7,8])
trainx,testx,trainy,testy=train_test_split(inp2,tar2,test_size=0.25)


print("\n","**"*20,"Linear Regression","**"*20)
le=LinearRegression()
le.fit(trainx,trainy)
letrainr=le.predict(trainx)
letestr=le.predict(testx)
print("#"*15,"Training","#"*15,"\t"*2,"  ","#"*15,"Testing","#"*15)
print("R2_Score  : ",r2_score(trainy,letrainr),"\t"*3,"    R2_Score  : ",r2_score(testy,letestr))



print("\n\n","**"*20,"Gradient Boosting Regressor","**"*20)
gbr=GradientBoostingRegressor(max_depth=6,max_leaf_nodes=32)
gbr.fit(trainx,trainy)
gbr_train=gbr.predict(trainx)
gbr_test=gbr.predict(testx)
print("#"*15,"Training","#"*15,"\t"*3,"    ","#"*15,"Testing","#"*15)
print("R2_Score : ",r2_score(trainy,gbr_train),"\t"*5,"      R2_Score : ",r2_score(testy,gbr_test))
