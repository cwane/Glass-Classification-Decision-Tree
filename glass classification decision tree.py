#Import all the necessary libraries
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import pandas as pd

#Load the dataset
df= pd.read_csv('glass.csv')
df

#Information of dataset
df.info()


df.head(214)

#get the dunny dataset
df2 = pd.get_dummies(df)

df2.head(10)


# Decision tree using entropy

clf = tree.DecisionTreeClassifier(criterion='entropy')


X = df2.iloc[:,:-1]
X.head()

y = df2.iloc[:, -1:]
y.head()


#.split the dataset such that 33% goes to test data and 67% goes to train data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)



X_train.head()



X_test.head()


#concatenate the feature columns and class column to visualize the complete training data
training_data =pd.concat([X_train,y_train],axis=1)


#observe training data
training_data.head()



# fit the decision tree model
clf = clf.fit(X_train,y_train)



#make predictions on test data
predicted = clf.predict(X_test)



y_test.head()



# Get the class distribution from the dataset
class_counts = df['Type'].value_counts()

# Create a bar plot for class distribution of glass classification dataset
plt.figure(figsize=(10, 6))
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution of Glass Classification Dataset')
plt.show()



# Define the feature names
feature_names = X.columns.tolist()

# Plot the decision tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=feature_names, class_names=True, filled=True, rounded=True)
plt.title("Decision Tree for Glass classification Dataset using entropy")
plt.show()

# know depth of decision tree
clf = DecisionTreeClassifier().fit(X_train, y_train)
tree_depth = clf.tree_.max_depth
print(f"The depth of the decision tree is: {tree_depth}")


#gives tree with all the textual information of the tree
tree.plot_tree(clf)


#Confusion matrix in text form

confusion_mat = confusion_matrix(y_test,predicted)
print("Confusion Matrix:")
print(confusion_mat)


#Heatmap of confusion matrix
sns.heatmap(conf_mat,annot=True)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()


#classification report of the decision tree
classification_report=classification_report(y_test, predicted)
print("Classification Report for Glass classification dataset using entropy:")

print(classification_report)



#Gives all the decision tree at the specified range of depth (optional)
# Initialize lists to store results
depths = []
confusion_matrices = []
classification_reports = []

# Define the range of depths to explore
min_depth = 1
max_depth = 10

# Iterate over different depths
for depth in range(min_depth, max_depth + 1):
    # Create and fit the decision tree model with entropy criterion
    dt_model = DecisionTreeClassifier(max_depth=depth, criterion='entropy')
    dt_model.fit(X_train, y_train)

    # Append the depth to the list
    depths.append(depth)

    # Plot the decision tree
    plt.figure(figsize=(10, 6))
    plot_tree(dt_model, feature_names=feature_names, filled=True, rounded=True, class_names=True)
    plt.title(f"Decision Tree (Depth {depth})")
    plt.show()

    # Make predictions on the test set
    y_pred = dt_model.predict(X_test)

    # Compute confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    # Append the confusion matrix and classification report to the respective lists
    confusion_matrices.append(cm)
    classification_reports.append(cr)

# Print the confusion matrix and classification report for each depth
for depth, cm, cr in zip(depths, confusion_matrices, classification_reports):
    print(f"Depth {depth}:")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)
    print("-------------------")



#Decision tree, confusion matrixa and classification report at depth 2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report

# Define the feature names
feature_names = X.columns.tolist()

# Initialize the list to store depths
depths = []

# Create and fit the decision tree model with depth 2 and entropy criterion
clf = DecisionTreeClassifier(max_depth=2, criterion='entropy')
clf.fit(X_train, y_train)

# Append the depth to the list
depths.append(2)

# Plot the decision tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=feature_names, filled=True, rounded=True, class_names=True)
plt.title("Decision Tree (Depth 2)")
plt.show()

# Create and fit the decision tree model with a specific depth
depth = 2
dt_model = DecisionTreeClassifier(max_depth=depth)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Compute the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title(f'Confusion Matrix (Depth {depth})')
plt.show()

print("Classification Report for depth 2")
classification_report = classification_report(y_test, y_pred)
print(classification_report)




#Decision tree, confusion matrixa and classification report at depth 3
# Define the feature names
feature_names = X.columns.tolist()

# Initialize the list to store depths
depths = []

# Create and fit the decision tree model with depth 2 and entropy criterion
clf = DecisionTreeClassifier(max_depth=3, criterion='entropy')
clf.fit(X_train, y_train)

# Append the depth to the list
depths.append(3)

# Plot the decision tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=feature_names, filled=True, rounded=True, class_names=True)
plt.title("Decision Tree (Depth 3)")
plt.show()

# Create and fit the decision tree model with a specific depth
depth = 3
dt_model = DecisionTreeClassifier(max_depth=depth)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Compute the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title(f'Confusion Matrix (Depth {depth})')
plt.show()

print("Classification Report for depth 3")
classification_report = classification_report(y_test, y_pred)
print(classification_report)




#Decision tree, confusion matrixa and classification report at depth 5
# Define the feature names
feature_names = X.columns.tolist()

# Initialize the list to store depths
depths = []

# Create and fit the decision tree model with depth 2 and entropy criterion
clf = DecisionTreeClassifier(max_depth=5, criterion='entropy')
clf.fit(X_train, y_train)

# Append the depth to the list
depths.append(5)

# Plot the decision tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=feature_names, filled=True, rounded=True, class_names=True)
plt.title("Decision Tree (Depth 5)")
plt.show()

# Create and fit the decision tree model with a specific depth
depth = 5
dt_model = DecisionTreeClassifier(max_depth=depth)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Compute the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title(f'Confusion Matrix (Depth {depth})')
plt.show()

print("Classification Report for depth 5")
classification_report = classification_report(y_test, y_pred)
print(classification_report)



# Using Gini index

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report

# Create and fit the decision tree model with Gini index criterion
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=feature_names, filled=True, rounded=True, class_names=True)
plt.title("Decision Tree for Glass Classification Dataset using Gini Index")
plt.show()

clf = DecisionTreeClassifier().fit(X_train, y_train)
tree_depth = clf.tree_.max_depth
print(f"The depth of the decision tree is: {tree_depth}")


confusion_mat = confusion_matrix(y_test,predicted)
print("Confusion Matrix:")
print(confusion_mat)

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(conf_mat,annot=True)
plt.title('Confusion Matrix for Glass Classification Dataset')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()


print("Classification Report:")
classification_report=classification_report(y_test, predicted)
print(classification_report)





#Decision tree, confusion matrixa and classification report at depth 2 using gini index
# Define the feature names
feature_names = X.columns.tolist()

# Initialize the list to store depths
depths = []

# Create and fit the decision tree model with depth 2 and entropy criterion
clf = DecisionTreeClassifier(max_depth=2, criterion='gini')
clf.fit(X_train, y_train)

# Append the depth to the list
depths.append(2)

# Plot the decision tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=feature_names, filled=True, rounded=True, class_names=True)
plt.title("Decision Tree using gini index (Depth 2)")
plt.show()

# Create and fit the decision tree model with a specific depth
depth = 2
dt_model = DecisionTreeClassifier(max_depth=depth)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Compute the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title(f'Confusion Matrix (Depth {depth})')
plt.show()

print("Classification Report for depth 2 using gini index ")
classification_report = classification_report(y_test, y_pred)
print(classification_report)


#Decision tree, confusion matrixa and classification report at depth 3 using gini index
# Define the feature names
feature_names = X.columns.tolist()

# Initialize the list to store depths
depths = []

# Create and fit the decision tree model with depth 2 and Gini index criterion
clf = DecisionTreeClassifier(max_depth=3, criterion='gini')
clf.fit(X_train, y_train)

# Append the depth to the list
depths.append(3)

# Plot the decision tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=feature_names, filled=True, rounded=True, class_names=True)
plt.title("Decision Tree using gini index (Depth 3)")
plt.show()

# Create and fit the decision tree model with a specific depth
depth = 3
dt_model = DecisionTreeClassifier(max_depth=depth)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Compute the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title(f'Confusion Matrix (Depth {depth})')
plt.show()

# Print the classification report
print("Classification Report for depth 3 using gini index")
cr = classification_report(y_test, y_pred)
print(cr)



#Decision tree, confusion matrixa and classification report at depth 5 using gini index
# Define the feature names
feature_names = X.columns.tolist()

# Initialize the list to store depths
depths = []

# Create and fit the decision tree model with depth 2 and Gini index criterion
clf = DecisionTreeClassifier(max_depth=5, criterion='gini')
clf.fit(X_train, y_train)

# Append the depth to the list
depths.append(5)

# Plot the decision tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=feature_names, filled=True, rounded=True, class_names=True)
plt.title("Decision Tree using gini index (Depth 5)")
plt.show()

# Create and fit the decision tree model with a specific depth
depth = 5
dt_model = DecisionTreeClassifier(max_depth=depth)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Compute the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title(f'Confusion Matrix (Depth {depth})')
plt.show()

# Print the classification report
print("Classification Report for depth 5 using gini index")
cr = classification_report(y_test, y_pred)
print(cr)

