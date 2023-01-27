import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#data loading
df= pd.read_csv('waypoints-withLabelsVol2.csv')
print(df)

pairplot=sns.pairplot(df,hue='label')
plt.plot()
plt.show()

print("hihi")
data=df.values
X = data[:,1:]
Y = data[:,0] #label

#chart
df.groupby(['label']).count().plot(kind='bar')
locs, labels = plt.xticks()
plt.xticks(rotation=0)
plt.title("Routes")
plt.xlabel("")
plt.show()


X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)



"""  TO SUPPORT VECTOR MACHINE BGAZEI KAKO ACCURACY
svn = SVC()
svn.fit(X_train,y_train)

#prediction from test dataset
predictions = svn.predict(X_test)

#Calculate the accuracy
acu_score = accuracy_score(y_test,predictions)

print(acu_score)
print(classification_report(y_test,predictions))
"""



#Decision tree
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)
acu_score_tree= accuracy_score(y_test,dtree_predictions)

print("Desicion tree accuracy score:",acu_score_tree)
print(classification_report(y_test,dtree_predictions))


#Naive bayes

gnb_model = GaussianNB().fit(X_train,y_train)
gnb_predictions = gnb_model.predict(X_test)
acu_score_bayes = accuracy_score(y_test,gnb_predictions)

print("Naive Bayes accuracy score:",acu_score_bayes)
print(classification_report(y_test,gnb_predictions))

#Knn
knn_model = KNeighborsClassifier(n_neighbors = 3).fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
acu_score_knn = accuracy_score(y_test,knn_predictions)
print("Knn accuracy score:",acu_score_knn)
print(classification_report(y_test,knn_predictions))

#Random Tree
rtree_model = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
rtree_predictions = rtree_model.predict(X_test)
acu_score_rtree = accuracy_score(y_test,rtree_predictions)
print("Random Tree accuracy score:",acu_score_rtree)
print(classification_report(y_test,rtree_predictions))


#Kfold decisionTree
K = 5
kf =KFold(n_splits = K, shuffle = True)
list_conf_mat = []
for train_idx, test_idx in kf.split(X):
    dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X[train_idx,:], Y[train_idx])
    predictions = dtree_model.predict(X[test_idx,:])
    list_conf_mat.append(confusion_matrix(y_true=Y[test_idx], y_pred=predictions))
for conf_mat in list_conf_mat:
    print("Tree K fold :\n",conf_mat)
    print('')

k_fold_average_tree=np.mean([(cm[0, 0] + cm[1, 1]) / sum(sum(cm)) for cm in list_conf_mat])


print("Decision tree 5-fold cross validation average: ",k_fold_average_tree)
print("Telos")






#arxeio ais
ais= pd.read_csv('ais_lat-lon.csv')
ais=ais.drop("mmsi",axis=1)
predictions = dtree_model.predict(ais)

print(predictions)







