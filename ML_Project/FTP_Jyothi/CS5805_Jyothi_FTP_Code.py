#%%----------Phase I: Feature Engineering & EDA--------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.decomposition import PCA,TruncatedSVD
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import f_classif
from prettytable import PrettyTable
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix,roc_auc_score, roc_curve, recall_score, precision_score, f1_score, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

#Data cleaning
df = pd.read_csv(r'/Users/jyothi/Documents/SEM2/Machine_Learning/Project/Airlines.csv')
print(df.columns)
print(df.head().to_string())
print("\nMissing values in the dataset")
print(df.isna().sum().sum() + df.isnull().sum().sum())

#Check for data duplications and removal
print("\nDuplicated values in the dataset")
print(df.duplicated().sum())
print("\nNumber of unique values in each feature of whole dataset")
print(df.nunique())

#Aggregation
df_dayofweek = df[['DayOfWeek', 'Delay']].groupby('DayOfWeek')['Delay'].sum().reset_index().sort_values(by='Delay', ascending=False)
print(df_dayofweek.to_string())
df_airline = df[['Airline', 'Delay']].groupby('Airline')['Delay'].sum().reset_index().sort_values(by='Delay', ascending=False)
print(df_airline.to_string())

#Down sampling
df = df.drop(['id', 'AirportTo'], axis=1)
df_airline = df['Airline'].value_counts().sort_values(ascending=False)
df = df[df['Airline'].isin(['WN', 'DL', 'OO', 'AA', 'MQ'])]
df_airport_to = df['AirportFrom'].value_counts().sort_values(ascending=False)
df = df[df['AirportFrom'].isin(['ATL', 'ORD', 'DFW', 'DEN', 'LAX'])]
df = df[::2]
print("\nThe shape of the dataset after downsampling:")
print(df.shape)

print("\nThe value counts of each class of target feature:")
print(df['Delay'].value_counts())

#Anomaly detection/Outlier Analysis and removal using IQR method
q1 = df['Length'].quantile(0.25)
q3 = df['Length'].quantile(0.75)
IQR = q3 - q1
lower_bound = q1 - 1.5 * IQR
upper_bound = q3 + 1.5 * IQR
df = df[(df['Length'] < upper_bound) & (df['Length'] > lower_bound)]
df_new = df[::]
sns.boxplot(data=df, x='Length')
plt.title("Box plot of Length feature after removal of outliers using IQR method")
plt.show()
df_apr = df[::]
#Discretization & Binarization: Label Encoding/one hot encoding
one_hot_enc_col = ['AirportFrom', 'Airline', 'DayOfWeek']
df = pd.get_dummies(df, columns=one_hot_enc_col, drop_first=True)
for i in df.columns:
    if i not in ['Length', 'Time', 'Delay', 'Flight']:
        df[i] = df[i].map({False:0, True:1})
print("\nOne hot encoded dataset")
print(df.head().to_string())
df_orig= df[::]
X_orig = df.drop('Delay', axis=1)
X_reg = df[::]

# Correlation matrix using heatmap
correlation = df.corr().round(2)
plt.figure(figsize = (18,11))
sns.heatmap(correlation, annot = True, cbar=True)
plt.title('Heatmap of correlation between all the features')
plt.show()

# Covariance matrix using heatmap
correlation = df.cov().round(2)
plt.figure(figsize = (18,11))
sns.heatmap(correlation, annot = True, cbar=True)
plt.title('Heatmap of covariance between all the features')
plt.show()

#Variable Transformation: Standardization
sc = StandardScaler()
for i in ['Flight', 'Length', 'Time']:
    df[i] = sc.fit_transform(df[[i]])
print("\nStandardized dataset")
print(df.head().to_string())
X = df.drop('Delay', axis=1)
y = df['Delay']
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=5805, shuffle=True,stratify=y)

# Dimensionality reduction
# Feature importance using Random forest analysis
# threshold of 95% and hence dropped features Airline_MQ, AirportFrom_DFW, AirportFrom_LAX, Airline_DL, AirportFrom_DEN, Airline_OO
model_rf = RandomForestClassifier(random_state=5805)
model_rf.fit(X, y)
importances = model_rf.feature_importances_
print("Feature importances from Random forest method")
indices = np.argsort(importances)
sortedImportance = importances[indices] * 100
print(sortedImportance[::-1].round(2))
sorted_features = X.columns[indices]
plt.figure(figsize=(15, 10))
plt.barh(range(len(sortedImportance)), sortedImportance, color='darkblue')  # Light blue hex code
plt.yticks(range(len(sorted_features)), sorted_features)
plt.title('Feature importances by RandomForestClassifier')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.show()

# Dimensionality reduction/Feature selection
# Using PCA
pca = PCA()
pca.fit(X)
cum_var = np.round(np.cumsum(sorted(pca.explained_variance_ratio_, reverse=True)) * 100, decimals=2)
print("Cumulative explained variance of each component from PCA")
print(cum_var)
labels = [i for i in range(1, len(cum_var) + 1)]
plt.plot(range(len(cum_var)), cum_var, alpha=0.9, color='darkblue')  # Dark blue line
x_point = 11
y_point = 95
plt.axvline(x=x_point, color='darkblue', linestyle='--')
plt.axhline(y=y_point, color='darkblue', linestyle='--')
plt.annotate(f'({x_point},{y_point})', (x_point, y_point), color='black', textcoords="offset points", xytext=(-15, 10), ha='center')
plt.xlabel("Number of components", color='black')
plt.ylabel("Percentage contribution to variance", color='black')
plt.title("Cumulative explained variance of each component from PCA", color='black')
plt.tick_params(axis='x', colors='black')
plt.tick_params(axis='y', colors='black')
plt.show()

# Dimensionality reduction/Feature selection
# Using Singluar Value Decomposition
tsvd = TruncatedSVD(n_components=17)
tsvd_result = tsvd.fit(X)
print("Explained variance ratio from SVD ", tsvd.explained_variance_ratio_.round(2))
print(f"Singular values of features {tsvd.singular_values_.round(2)}")
cum_evr = np.cumsum(100 * tsvd.explained_variance_ratio_)
labels = [i for i in range(1, len(cum_evr) + 1)]
plt.bar(x=labels, height=cum_evr, alpha=0.8, color='darkblue')
x_point = 12
y_point = 95
plt.axvline(x=x_point, linestyle='--', color='red')
plt.axhline(y=y_point, linestyle='--', color='red')
plt.annotate(f'({x_point},{y_point})', (x_point, y_point), textcoords="offset points", xytext=(-15,10), color='black')
plt.xlabel("Number of components", color='black')
plt.ylabel("Percentage contribution to variance", color='black')
plt.title("Cumulative variance ratio of each component from SVD", color='black')
plt.xticks(labels)
plt.show()


# Dimensionality reduction/Feature selection
# Using Variance Inflation Factor
print("VIF analysis")
vif_fea = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
for i,j in zip(df.columns, vif_fea):
    print(i,round(j,2))
_, S, _ = np.linalg.svd(X, full_matrices=False)
print("Singular values of the dataset")
print(S.round(2))
condition_number = np.linalg.cond(X_train)
print("Condition number : ", round(condition_number,2))

# Data imbalance analysis
sns.set_style('whitegrid')
sns.countplot(data=df,x='Delay')
plt.title('Countplot of # of obs. for each class of Delay')
plt.xlabel("Whether flight delayed")
plt.ylabel("# of samples")
plt.show()

#%%----------Phase II: Regression Analysis--------
df.drop(['Airline_MQ', 'AirportFrom_ORD', 'AirportFrom_DFW', 'Airline_DL','AirportFrom_LAX', 'AirportFrom_DEN', 'Airline_OO'], axis=1, inplace=True)
mean_length = df_orig['Length'].mean()
std_length = df_orig['Length'].std()
sc=StandardScaler()
for i in ['Flight', 'Time', 'Length']:
    X_reg[i] = sc.fit_transform(X_reg[[i]])
y_reg = X_reg['Length']
y_reg = sc.fit_transform(y_reg.values.reshape(-1,1))
X_reg = X_reg.drop('Length', axis=1)
# X_reg = sm.add_constant(X_reg)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg,test_size=0.2, random_state=5805, shuffle=True)
print(df_orig.head().to_string())

#F-test analysis
f_test, p_values = f_classif(X_reg, y_reg)
pt = PrettyTable()
pt.field_names = ['Feature', 'F-test score', 'p-value']
for i in range(len(f_test)):
    pt.add_row([X.columns[i], round(f_test[i],2), round(p_values[i],2)])
print(pt)
# Prediction with Linear Regression
model_linear = LinearRegression()
X_reg_train = sm.add_constant(X_reg_train)
X_reg_test = sm.add_constant(X_reg_test)
model_linear.fit(X_reg_train, y_reg_train)
y_pred_train = model_linear.predict(X_reg_train)
y_pred_train_orig = y_pred_train*std_length+mean_length
y_pred_test = model_linear.predict(X_reg_test)
y_pred_test_orig = y_pred_test*std_length+mean_length
print("The coefficients of the Linear Regression model")
print("Feature", "Coefficient")
for i in range(len(X_reg_train.columns)):
    print(X_reg_train.columns[i], round(model_linear.coef_[0][i],2))
mse = mean_squared_error(y_reg_train*std_length+mean_length, y_pred_train_orig)
print(f"Mean Squared Error for training set (MSE): {mse:.3f}")
pt=PrettyTable()
pt.field_names=['Model', 'R-squared', 'Adj. R-squared', 'AIC', 'BIC', 'Mean Squared Error']
r_squared = model_linear.score(X_reg_train, y_reg_train)
n = len(y_reg_train)
p = X_reg_train.shape[1] - 1
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
aic = n*np.log(mse) + 2*p
bic = n*np.log(mse) + p*np.log(n)
pt.add_row(['Linear Regression', round(r_squared,2), round(adjusted_r_squared,2),
round(aic,2), round(bic,2), round(mse,2)])
print(pt)
index=np.arange(len(y_reg_test))
plt.title("Prediction with Linear Regression")
plt.plot(index, y_reg_test * std_length + mean_length, color='blue', label='Actual values of test set')  # Blue for actual values
plt.plot(index, y_pred_test_orig, color='lightyellow', label='Predicted values with test set')  # Yellow for predicted values
plt.xlabel("# of samples")
plt.ylabel("Length")
plt.legend()
plt.show()

model_ols = sm.OLS(y_reg_train, X_reg_train).fit()
y_pred_train = model_ols.predict(X_reg_train)
y_pred_train_orig = y_pred_train*std_length+mean_length
y_pred_test = model_ols.predict(X_reg_test)
y_pred_test_orig = y_pred_test*std_length+mean_length
print(model_ols.summary())

#T-test analysis
pt_ttest = PrettyTable()
pt_ttest.field_names = ['Feature', 't-statistic', 'p-value']
print("t-test analysis")
for i in range(len(X_reg_train.columns)):
    pt_ttest.add_row([X_reg_train.columns[i], round(model_ols.tvalues[i],2),round(model_ols.pvalues[i],2)])
print(pt_ttest)

eliminated_features = ['DayofWeek_3','DayofWeek_4','DayofWeek_5','DayofWeek_2','DayOfWeek_7','Flight','DayOfWeek_6']
## Backward stepwise regression
print("Eliminating the feature with highest p-value which is DayOfWeek_3")
X_reg_train = X_reg_train.drop('DayOfWeek_3', axis=1)
X_reg_test = X_reg_test.drop('DayOfWeek_3', axis=1)
model_ols = sm.OLS(y_reg_train, X_reg_train).fit()
y_pred_train = model_ols.predict(X_reg_train)
y_pred_train_orig = y_pred_train*std_length+mean_length
y_pred_test = model_ols.predict(X_reg_test)
y_pred_test_orig = y_pred_test*std_length+mean_length
print(model_ols.summary())

print("Eliminating DayOfWeek_4")
X_reg_train = X_reg_train.drop('DayOfWeek_4', axis=1)
X_reg_test = X_reg_test.drop('DayOfWeek_4', axis=1)
model_ols = sm.OLS(y_reg_train, X_reg_train).fit()
y_pred_train = model_ols.predict(X_reg_train)
y_pred_train_orig = y_pred_train*std_length+mean_length
y_pred_test = model_ols.predict(X_reg_test)
y_pred_test_orig = y_pred_test*std_length+mean_length
print(model_ols.summary())

print("Eliminating DayOfWeek_5")
X_reg_train = X_reg_train.drop('DayOfWeek_5', axis=1)
X_reg_test = X_reg_test.drop('DayOfWeek_5', axis=1)
model_ols = sm.OLS(y_reg_train, X_reg_train).fit()
y_pred_train = model_ols.predict(X_reg_train)
y_pred_train_orig = y_pred_train*std_length+mean_length
y_pred_test = model_ols.predict(X_reg_test)
y_pred_test_orig = y_pred_test*std_length+mean_length
print(model_ols.summary())

print("Eliminating DayOfWeek_2")
X_reg_train = X_reg_train.drop('DayOfWeek_2', axis=1)
X_reg_test = X_reg_test.drop('DayOfWeek_2', axis=1)
model_ols = sm.OLS(y_reg_train, X_reg_train).fit()
y_pred_train = model_ols.predict(X_reg_train)
y_pred_train_orig = y_pred_train*std_length+mean_length
y_pred_test = model_ols.predict(X_reg_test)
y_pred_test_orig = y_pred_test*std_length+mean_length
print(model_ols.summary())

print("Eliminating DayOfWeek_7")
X_reg_train = X_reg_train.drop('DayOfWeek_7', axis=1)
X_reg_test = X_reg_test.drop('DayOfWeek_7', axis=1)
model_ols = sm.OLS(y_reg_train, X_reg_train).fit()
y_pred_train = model_ols.predict(X_reg_train)
y_pred_train_orig = y_pred_train*std_length+mean_length
y_pred_test = model_ols.predict(X_reg_test)
y_pred_test_orig = y_pred_test*std_length+mean_length
print(model_ols.summary())

print("Eliminating Flight")
X_reg_train = X_reg_train.drop('Flight', axis=1)
X_reg_test = X_reg_test.drop('Flight', axis=1)
model_ols = sm.OLS(y_reg_train, X_reg_train).fit()
y_pred_train = model_ols.predict(X_reg_train)
y_pred_train_orig = y_pred_train*std_length+mean_length
y_pred_test = model_ols.predict(X_reg_test)
y_pred_test_orig = y_pred_test*std_length+mean_length
print(model_ols.summary())

print("Eliminating DayOfWeek_6")
X_reg_train = X_reg_train.drop('DayOfWeek_6', axis=1)
X_reg_test = X_reg_test.drop('DayOfWeek_6', axis=1)
model_ols = sm.OLS(y_reg_train, X_reg_train).fit()
y_pred_train = model_ols.predict(X_reg_train)
y_pred_train_orig = y_pred_train*std_length+mean_length
y_pred_test = model_ols.predict(X_reg_test)
y_pred_test_orig = y_pred_test*std_length+mean_length
print(model_ols.summary())

eliminated_features = ['DayofWeek_3','DayofWeek_4','DayofWeek_5','DayofWeek_2','DayOfWeek_7','Flight','DayOfWeek_6']
print(f"eliminated features: {eliminated_features}")

#condition number of the remaining features of the dataset after backward stepwise regression
print("Condition number of the remaining features of the dataset after backward stepwise regression")
print(round(np.linalg.cond(X_reg_train),2))
print("Singular values of the remaining features of the dataset after backward stepwise regression")
_, S, _ = np.linalg.svd(X_reg_train, full_matrices=False)
print(np.round(S, 2))

# train, test and predicted values in one plot
index=np.arange(len(y_reg_test[:100]))
plt.title("Plot of train, test and predicted values of backward stepwise regression")
plt.plot(index,y_reg_train[0:100]*std_length+mean_length)
plt.plot(index,y_reg_test[0:100]*std_length+mean_length)
plt.plot(index,y_pred_test_orig[0:100])
plt.xlabel("# of samples")
plt.ylabel("Length")
plt.legend(['Actual values of train set', 'Actual values of test set', 'Predicted values with test set'])
plt.show()

print("The coefficients of the OLS model using the final set of features after backward stepwise regression")
print("Feature", "Coefficient")
for i in range(len(X_reg_train.columns)):
    print(X_reg_train.columns[i], round(model_ols.params[i],2))
mse = mean_squared_error(y_reg_train*std_length+mean_length, y_pred_train_orig)
print("Mean Squared Error for training set(MSE):", round(mse,2))
pt2 = PrettyTable()
pt2.field_names = ['Model', 'R-squared', 'Adj. R-squared', 'AIC', 'BIC', 'Mean Squared Error']
r_squared = model_ols.rsquared
n = len(y_reg_train)
p = X_reg_train.shape[1] - 1
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
aic = model_ols.aic
bic = model_ols.bic
mse = mean_squared_error(y_reg_test*std_length+mean_length, y_pred_test_orig)
# print("Mean Squared Error for testing set(MSE):", mse)
pt2.add_row(['Backward Stepwise Regression', round(r_squared,2),
round(adjusted_r_squared,2), round(aic,2), round(bic,2), round(mse,2)])
pt.add_row(['Backward Stepwise Regression', round(r_squared,2),
round(adjusted_r_squared,2), round(aic,2), round(bic,2), round(mse,2)])
print(pt2)
print(pt)

#Confidence interval analysis
index=np.arange(len(y_reg_test))
y_prediction = model_ols.get_prediction(X_reg_test)
conf_intr = model_ols.conf_int()
ci_pred_frame=y_prediction.summary_frame(alpha=0.05)
ci_upper = ci_pred_frame.obs_ci_upper
ci_lower = ci_pred_frame.obs_ci_lower
plt.plot(index,y_pred_test_orig,label="Predicted",color='blue')
plt.fill_between(index, ci_upper*std_length+mean_length,
ci_lower*std_length+mean_length, color='blue', alpha=0.3, label="Confidence Interval")
plt.xlabel("Number of observations")
plt.ylabel("Sales")
plt.title("Confidence interval of Predicted values summary_frame")
plt.legend()
plt.show()
model_ols_sr = sm.OLS(y_reg_train, X_reg_train).fit()
print(model_ols_sr.summary())
#%%%--------Phase III: Classification Analysis----------------

## Pre-prune decision tree
X = df.drop('Delay', axis=1)
print(X.columns)
y = df['Delay']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=5805, shuffle=True)
pt = PrettyTable()
pt.field_names = ['Classifier', 'Accuracy', 'Confusion Matrix', 'Precision','Sensitivity or Recall', 'Specificity', 'F-score', 'AUC']
clf=DecisionTreeClassifier(random_state=5805)
tuned_parameters = [{'max_depth':[5,7,9,11],
'min_samples_split': [2,5,10,15],
'min_samples_leaf':[1,2,4,8],
'max_features':[2,4,7],
'splitter':['best','random'],
'criterion':['gini','entropy','log_loss']}]
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)
grid_search = GridSearchCV(clf, tuned_parameters, cv=stratified_kfold,scoring='accuracy')
grid_search.fit(X, y)
print("Best Hyperparameters:", grid_search.best_params_)
best_classifier = grid_search.best_estimator_
y_train_pred = best_classifier.predict(X_train)
y_test_pred = best_classifier.predict(X_test)
pre_acc = accuracy_score(y_test, y_test_pred)
print(f'Train accuracy of pre pruned decision tree {round(accuracy_score(y_test,y_test_pred),2)}')
print(f'Test accuracy of pre pruned decision tree {round(pre_acc, 2)}')
conf_matrix_pre = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = conf_matrix_pre.ravel()
specificity_pre = tn/(tn+fp)
y_prob_pre = best_classifier.predict_proba(X_test)[:, 1] # Probability of the positive class
roc_auc_pre = roc_auc_score(y_test, y_prob_pre)
recall_pre = recall_score(y_test, y_test_pred)
precision_pre = precision_score(y_test, y_test_pred)
f1_pre = f1_score(y_test, y_test_pred)
# heatmap of confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_pre, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix of Pre pruned decision tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
pt.add_row(['Pre pruned Decision Tree Classifier', round(pre_acc,2),
conf_matrix_pre.round(2), round(precision_pre,2), round(recall_pre,2),
round(specificity_pre,2), round(f1_pre,2), round(roc_auc_pre,2)])
fpr_pre, tpr_pre, thresholds_pre = roc_curve(y_test, y_prob_pre)
plt.figure(figsize=(8, 6))
plt.plot(fpr_pre, tpr_pre, linewidth=2, label="ROC curve of Pre pruned decision tree (area = {:.2f})".format(roc_auc_pre))
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve ')
plt.legend(loc="lower right")
plt.show()
print(pt)
tree.plot_tree(best_classifier)
plt.show()


## Post pruned decision tree
fold_accuracies = []
fold_train_accuracies = []
path = best_classifier.cost_complexity_pruning_path(X_train,y_train)
alphas = path['ccp_alphas']

# Iterate over different values of ccp_alpha
for alpha in alphas:
    clf = DecisionTreeClassifier(random_state=5805, ccp_alpha=alpha)
    clf.fit(X_train, y_train)
    accuracy_test = clf.score(X_test, y_test)
    fold_accuracies.append(accuracy_test)
    accuracy_train = clf.score(X_train, y_train)
    fold_train_accuracies.append(accuracy_train)

best_alpha_index = np.argmax(fold_accuracies)
best_alpha = alphas[best_alpha_index]
print(f"Best alpha value of post pruned decision tree: {best_alpha}")

# Plot accuracy vs. alpha for train and test sets
fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(alphas, fold_train_accuracies, marker="o", label="Train Accuracy", drawstyle="steps-post")
ax.plot(alphas, fold_accuracies, marker="o", label="Test Accuracy", drawstyle="steps-post")
ax.legend()
plt.grid()
plt.tight_layout()
plt.show()
best_alpha = alphas[np.argmax(accuracy_test)]
print("Best alpha value of post pruned decision tree", best_alpha)

clf = DecisionTreeClassifier(random_state=5805, ccp_alpha=best_alpha)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
post_acc = accuracy_score(y_test, y_test_pred)
conf_matrix_post = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = conf_matrix_pre.ravel()
specificity_post = tn/(tn+fp)
y_prob_post = clf.predict_proba(X_test)[:, 1] # Probability of the positive class
roc_auc_post = roc_auc_score(y_test, y_prob_post)
recall_post = recall_score(y_test, y_test_pred)
precision_post = precision_score(y_test, y_test_pred)
f1_post = f1_score(y_test, y_test_pred)
train_accuracy = round(accuracy_score(y_train, y_train_pred), 2)
print(f"Train accuracy of post pruned tree: {train_accuracy}")
test_accuracy = round(accuracy_score(y_test, y_test_pred), 2)
print(f"Train accuracy of post pruned tree: {test_accuracy}")
pt2 = PrettyTable()
pt2.field_names = ['Classifier', 'Accuracy', 'Confusion Matrix', 'Precision','Sensitivity or Recall', 'Specificity', 'F-score', 'AUC']
pt2.add_row(['Post pruned Decision Tree Classifier', round(post_acc,2),
conf_matrix_post.round(2), round(precision_post,2), round(recall_post,2),
round(specificity_post,2), round(f1_post,2), round(roc_auc_post,2)])
print(pt2)

# heatmap of confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_post, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix of Post pruned decision tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
pt.add_row(['Post pruned Decision Tree Classifier', round(post_acc,2),
conf_matrix_post.round(2), round(precision_post,2), round(recall_post,2),
round(specificity_post,2), round(f1_post,2), round(roc_auc_post,2)])
fpr_post, tpr_post, thresholds_post = roc_curve(y_test, y_prob_post)
plt.figure(figsize=(8, 6))
plt.plot(fpr_post, tpr_post, linewidth=2, label="ROC curve of Post pruned tree (area = {:.2f})".format(roc_auc_post))
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve ')
plt.legend(loc="lower right")
plt.show()
tree.plot_tree(clf)
plt.show()


## Logistic Regression
param_grid = {
'C': [0.1, 1.0, 10.0],
'penalty': ['l1', 'l2'],
'solver': ['liblinear', 'saga']
}
model_log = LogisticRegression(random_state=5805, max_iter=400)
grid_search = GridSearchCV(model_log, param_grid, cv=stratified_kfold,
scoring='accuracy')
grid_search.fit(X, y)
print("Best Hyperparameters:", grid_search.best_params_)
model_log = grid_search.best_estimator_
model_log.fit(X_train, y_train)
y_train_pred = model_log.predict(X_train)
y_test_pred = model_log.predict(X_test)
log_acc_train = accuracy_score(y_train, y_train_pred)
log_acc_test = accuracy_score(y_test, y_test_pred)
print(f'Train accuracy of Logistic Regression {round(accuracy_score(y_train,y_train_pred),2)}')
print(f'Test accuracy of Logistic Regression {round(log_acc_test,2)}')
conf_matrix_log = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = conf_matrix_log.ravel()
specificity_log = tn/(tn+fp)
y_prob_log = model_log.predict_proba(X_test)[:, 1]
roc_auc_log = roc_auc_score(y_test, y_prob_log)
recall_log = recall_score(y_test, y_test_pred)
precision_log = precision_score(y_test, y_test_pred)
f1_log = f1_score(y_test, y_test_pred)
pt2 = PrettyTable()
pt2.field_names = ['Classifier', 'Accuracy', 'Confusion Matrix', 'Precision',
'Sensitivity or Recall', 'Specificity', 'F-score', 'AUC']
pt2.add_row(['Logistic Regression', round(np.mean(log_acc_train),2),
conf_matrix_log.round(2), round(np.mean(precision_log),2),
round(np.mean(recall_log),2), round(np.mean(specificity_log),2),
round(np.mean(f1_log),2), round(np.mean(roc_auc_log),2)])
print(pt2)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_log, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix of Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
pt.add_row(['Logistic Regression', round(np.mean(log_acc_train),2),
conf_matrix_log.round(2), round(np.mean(precision_log),2),
round(np.mean(recall_log),2), round(np.mean(specificity_log),2),
round(np.mean(f1_log),2), round(np.mean(roc_auc_log),2)])
fpr_log, tpr_log, thresholds_log = roc_curve(y_test, y_prob_log)
plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, linewidth=2, label="ROC curve of Logistic Regression (area = {:.2f})".format(roc_auc_log))
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve ')
plt.legend(loc="lower right")
plt.show()

## KNN Classifier
param_grid = {'n_neighbors': [6, 9, 12, 15, 17, 19, 25, 29, 33, 37, 41, 43, 45, 49, 53, 57,61],
}
model_knn = KNeighborsClassifier()
grid_search = GridSearchCV(model_knn, param_grid, cv=stratified_kfold,scoring='accuracy')
grid_search.fit(X, y)
results = grid_search.cv_results_
k_values = results['param_n_neighbors'].data
mean_test_scores = results['mean_test_score']
error_rate = 1-mean_test_scores
plt.plot(k_values, error_rate)
plt.xlabel('K')
plt.ylabel('Error rate')
plt.title('Error rate vs K')
plt.show()
print("Best Hyperparameters:", grid_search.best_params_)

model_knn = grid_search.best_estimator_
model_knn.fit(X_train, y_train)
y_train_pred = model_knn.predict(X_train)
y_test_pred = model_knn.predict(X_test)
knn_acc_train = accuracy_score(y_train, y_train_pred)
knn_acc_test = accuracy_score(y_test, y_test_pred)
print(f'Train accuracy of KNN {round(accuracy_score(y_train, y_train_pred),2)}')
print(f'Test accuracy of KNN {round(knn_acc_test,2)}')

conf_matrix_knn = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = conf_matrix_knn.ravel()
specificity_knn = tn/(tn+fp)
y_prob_knn = model_knn.predict_proba(X_test)[:, 1] # Probability of the positive class
roc_auc_knn = roc_auc_score(y_test, y_prob_knn)
recall_knn = recall_score(y_test, y_test_pred)
precision_knn = precision_score(y_test, y_test_pred)
f1_knn = f1_score(y_test, y_test_pred)
pt2 = PrettyTable()
pt2.field_names = ['Classifier', 'Accuracy', 'Confusion Matrix', 'Precision','Sensitivity or Recall', 'Specificity', 'F-score', 'AUC']
pt2.add_row(['KNN', round(np.mean(knn_acc_train),2), conf_matrix_knn.round(2),
round(np.mean(precision_knn),2), round(np.mean(recall_knn),2),
round(np.mean(specificity_knn),2), round(np.mean(f1_knn),2),
round(np.mean(roc_auc_knn),2)])
print(pt2)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix of KNN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
pt.add_row(['KNN', round(np.mean(knn_acc_train),2), conf_matrix_knn.round(2),
round(np.mean(precision_knn),2), round(np.mean(recall_knn),2),
round(np.mean(specificity_knn),2), round(np.mean(f1_knn),2),
round(np.mean(roc_auc_knn),2)])
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_prob_knn)
plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, linewidth=2, label="ROC curve of KNN (area ={:.2f})".format(roc_auc_knn))
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve ')
plt.legend(loc="lower right")
plt.show()

## SVM Classifier
parameters = {
'kernel': ['linear', 'rbf', 'poly'], # Kernel types to try
}
# Perform grid search with cross-validation
svm_classifier = SVC(probability=True)
grid_search = GridSearchCV(svm_classifier, parameters, cv=stratified_kfold,n_jobs=-1)
grid_search.fit(X, y)
print("Best Hyperparameters:", grid_search.best_params_)

svm_lr = grid_search.best_estimator_
svm_lr.fit(X_train, y_train)
y_train_pred = svm_lr.predict(X_train)
y_test_pred = svm_lr.predict(X_test)
svm_lr_acc = accuracy_score(y_test, y_test_pred)
print(f'Train accuracy of SVM with best hyperparameters {round(np.mean(svm_lr_acc),2)}')
print(f'Test accuracy of SVM with best hyperparameters {round(accuracy_score(y_test, y_test_pred),2)}')

conf_matrix_svm_lr = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = conf_matrix_svm_lr.ravel()
specificity_svm_lr= tn/(tn+fp)
y_prob_svm_lr = svm_lr.predict_proba(X_test)[:, 1]
roc_auc_svm_lr = roc_auc_score(y_test, y_prob_svm_lr)
recall_svm_lr = recall_score(y_test, y_test_pred)
precision_svm_lr = precision_score(y_test, y_test_pred)
f1_svm_lr = f1_score(y_test, y_test_pred)
pt2 = PrettyTable()
pt2.field_names = ['Classifier', 'Accuracy', 'Confusion Matrix', 'Precision','Sensitivity or Recall', 'Specificity', 'F-score', 'AUC']
pt2.add_row(['SVM', round(np.mean(svm_lr_acc),2), conf_matrix_svm_lr.round(2),
round(np.mean(precision_svm_lr),2), round(np.mean(recall_svm_lr),2),
round(np.mean(specificity_svm_lr),2), round(np.mean(f1_svm_lr),2),
round(np.mean(roc_auc_svm_lr),2)])
print(pt2)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm_lr, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix of SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
pt.add_row(['SVM', round(np.mean(svm_lr_acc),2), conf_matrix_svm_lr.round(2),
round(np.mean(precision_svm_lr),2), round(np.mean(recall_svm_lr),2),
round(np.mean(specificity_svm_lr),2), round(np.mean(f1_svm_lr),2),
round(np.mean(roc_auc_svm_lr),2)])
fpr_svm_lr, tpr_svm_lr, thresholds_svm_lr = roc_curve(y_test, y_prob_svm_lr)
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm_lr, tpr_svm_lr, linewidth=2, label="ROC curve of SVM with best hyperparameters area = {:.2f})".format(roc_auc_svm_lr))
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve ')
plt.legend(loc="lower right")
plt.show()

## Naive Bayes Classifier
naive_bayes = GaussianNB()
train_scores = []
test_scores = []
for train_index, test_index in stratified_kfold.split(X, y):
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
    naive_bayes.fit(X_train_cv, y_train_cv)
train_score = naive_bayes.score(X_train_cv, y_train_cv)
train_scores.append(train_score)
test_score = naive_bayes.score(X_test_cv, y_test_cv)
test_scores.append(test_score)
print(f'Train accuracy of Gaussian Naive Bayes {round(np.mean(train_scores),2)}')
naive_bayes.fit(X_train, y_train)
y_train_pred = naive_bayes.predict(X_train)
y_test_pred = naive_bayes.predict(X_test)
print(f'Test accuracy of Gaussian Naive Bayes {round(np.mean(test_scores),2)}')
conf_matrix_nb = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = conf_matrix_nb.ravel()
specificity_nb = tn/(tn+fp)
y_prob_nb = naive_bayes.predict_proba(X_test)[:, 1]
roc_auc_nb = roc_auc_score(y_test, y_prob_nb)
recall_nb = recall_score(y_test, y_test_pred)
precision_nb = precision_score(y_test, y_test_pred)
f1_nb = f1_score(y_test, y_test_pred)
pt2 = PrettyTable()
pt2.field_names = ['Classifier', 'Accuracy', 'Confusion Matrix', 'Precision','Sensitivity or Recall', 'Specificity', 'F-score', 'AUC']
pt2.add_row(['Gaussian Naive Bayes', round(np.mean(test_scores),2),
conf_matrix_nb.round(2), round(np.mean(precision_nb),2),
round(np.mean(recall_nb),2), round(np.mean(specificity_nb),2),
round(np.mean(f1_nb),2), round(np.mean(roc_auc_nb),2)])
print(pt2)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_nb, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix of Gaussian Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
pt.add_row(['Gaussian Naive Bayes', round(np.mean(test_scores),2),
conf_matrix_nb.round(2), round(np.mean(precision_nb),2),
round(np.mean(recall_nb),2), round(np.mean(specificity_nb),2),
round(np.mean(f1_nb),2), round(np.mean(roc_auc_nb),2)])
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)
plt.figure(figsize=(8, 6))
plt.plot(fpr_nb, tpr_nb, linewidth=2, label="ROC curve of Gaussian Naive Bayes (area = {:.2f})".format(roc_auc_nb))
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

## MLP Classifier
mlp = MLPClassifier(max_iter=400, random_state=5805)
param_grid = {'hidden_layer_sizes': [(5, 5), (5, 10), (10, 10), (15, 15), (20, 20)]}
grid_search = GridSearchCV(mlp, param_grid, cv=stratified_kfold,scoring='accuracy')
grid_search.fit(X, y)
print("Best Hyperparameters:", grid_search.best_params_)

mlp = grid_search.best_estimator_
mlp.fit(X_train, y_train)
y_train_pred = mlp.predict(X_train)
y_test_pred = mlp.predict(X_test)
mlp_acc = accuracy_score(y_test, y_test_pred)
print(f'Train accuracy of Multilayer Perceptron with best hyperparameters {round(np.mean(mlp_acc),2)}')
print(f'Test accuracy of Multilayer Perceptron with best hyperparameters {round(accuracy_score(y_test, y_test_pred),2)}')

conf_matrix_mlp = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = conf_matrix_mlp.ravel()
specificity_mlp= tn/(tn+fp)
y_prob_mlp = mlp.predict_proba(X_test)[:, 1]
roc_auc_mlp = roc_auc_score(y_test, y_prob_mlp)
recall_mlp= recall_score(y_test, y_test_pred)
precision_mlp = precision_score(y_test, y_test_pred)
f1_mlp = f1_score(y_test, y_test_pred)
pt2 = PrettyTable()
pt2.field_names = ['Classifier', 'Accuracy', 'Confusion Matrix', 'Precision','Sensitivity or Recall', 'Specificity', 'F-score', 'AUC']
pt2.add_row(['Multilayer perceptron', round(np.mean(mlp_acc),2),
conf_matrix_mlp.round(2), round(np.mean(precision_mlp),2),
round(np.mean(recall_mlp),2), round(np.mean(specificity_mlp),2),
round(np.mean(f1_mlp),2), round(np.mean(roc_auc_mlp),2)])
print(pt2)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_mlp, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix of Multilayer perceptron')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
pt.add_row(['Multilayer perceptron', round(accuracy_score(y_test, y_test_pred),2),
conf_matrix_mlp.round(2), round(np.mean(precision_mlp),2),
round(np.mean(recall_mlp),2), round(np.mean(specificity_mlp),2),
round(np.mean(f1_mlp),2), round(np.mean(roc_auc_mlp),2)])
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_prob_mlp)
plt.figure(figsize=(8, 6))
plt.plot(fpr_mlp, tpr_mlp, linewidth=2, label="ROC curve of Multilayer perceptron (area = {:.2f})".format(roc_auc_mlp))
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve ')
plt.legend(loc="lower right")
plt.show()

##Comparing all above Classifier models
from prettytable import PrettyTable
pt_comparison = PrettyTable()
pt_comparison.field_names = ["Classifier", "Accuracy", "Confusion Matrix", "Precision", "Sensitivity or Recall", "Specificity", "F-Score", "AUC"]
pt_comparison.add_row(["Pre pruned Decision Tree Classifier", round(pre_acc, 2),
                       f"[[{conf_matrix_pre[0][0]} {conf_matrix_pre[0][1]}],\n [{conf_matrix_pre[1][0]} {conf_matrix_pre[1][1]}]]",
                       round(precision_pre, 2), round(recall_pre, 2),
                       round(specificity_pre, 2), round(f1_pre, 2), round(roc_auc_pre, 2)])

pt_comparison.add_row(["Post pruned Decision Tree Classifier", round(post_acc, 2),
                       f"[[{conf_matrix_post[0][0]} {conf_matrix_post[0][1]}],\n [{conf_matrix_post[1][0]} {conf_matrix_post[1][1]}]]",
                       round(precision_post, 2), round(recall_post, 2),
                       round(specificity_post, 2), round(f1_post, 2), round(roc_auc_post, 2)])

pt_comparison.add_row(["Logistic Regression", round(log_acc_test, 2),
                       f"[[{conf_matrix_log[0][0]} {conf_matrix_log[0][1]}],\n [{conf_matrix_log[1][0]} {conf_matrix_log[1][1]}]]",
                       round(precision_log, 2), round(recall_log, 2),
                       round(specificity_log, 2), round(f1_log, 2), round(roc_auc_log, 2)])

pt_comparison.add_row(["KNN", round(knn_acc_test, 2),
                       f"[[{conf_matrix_knn[0][0]} {conf_matrix_knn[0][1]}],\n [{conf_matrix_knn[1][0]} {conf_matrix_knn[1][1]}]]",
                       round(precision_knn, 2), round(recall_knn, 2),
                       round(specificity_knn, 2), round(f1_knn, 2), round(roc_auc_knn, 2)])

pt_comparison.add_row(["SVM", round(svm_lr_acc, 2),
                       f"[[{conf_matrix_svm_lr[0][0]} {conf_matrix_svm_lr[0][1]}],\n [{conf_matrix_svm_lr[1][0]} {conf_matrix_svm_lr[1][1]}]]",
                       round(precision_svm_lr, 2), round(recall_svm_lr, 2),
                       round(specificity_svm_lr, 2), round(f1_svm_lr, 2), round(roc_auc_svm_lr, 2)])

pt_comparison.add_row(["Gaussian Naive Bayes", round(np.mean(test_scores), 2),
                       f"[[{conf_matrix_nb[0][0]} {conf_matrix_nb[0][1]}],\n [{conf_matrix_nb[1][0]} {conf_matrix_nb[1][1]}]]",
                       round(precision_nb, 2), round(recall_nb, 2),
                       round(specificity_nb, 2), round(f1_nb, 2), round(roc_auc_nb, 2)])

pt_comparison.add_row(["Multilayer Perceptron", round(mlp_acc, 2),
                       f"[[{conf_matrix_mlp[0][0]} {conf_matrix_mlp[0][1]}],\n [{conf_matrix_mlp[1][0]} {conf_matrix_mlp[1][1]}]]",
                       round(precision_mlp, 2), round(recall_mlp, 2),
                       round(specificity_mlp, 2), round(f1_mlp, 2), round(roc_auc_mlp, 2)])
print(pt_comparison)
plt.figure(figsize=(8, 6))
plt.plot(fpr_pre, tpr_pre, linewidth=2, label="ROC curve of Pre pruned decision tree (area = {:.2f})".format(roc_auc_pre))
plt.plot(fpr_post, tpr_post, linewidth=2, label="ROC curve of Post pruned tree (area = {:.2f})".format(roc_auc_post))
plt.plot(fpr_log, tpr_log, linewidth=2, label="ROC curve of Logistic Regression(area = {:.2f})".format(roc_auc_log))
plt.plot(fpr_knn, tpr_knn, linewidth=2, label="ROC curve of KNN (area = {:.2f})".format(roc_auc_knn))
plt.plot(fpr_svm_lr, tpr_svm_lr, linewidth=2, label="ROC curve of SVM (area = {:.2f})".format(roc_auc_svm_lr))
plt.plot(fpr_nb, tpr_nb, linewidth=2, label="ROC curve of Gaussian Naive Bayes (area = {:.2f})".format(roc_auc_nb))
plt.plot(fpr_mlp, tpr_mlp, linewidth=2, label="ROC curve of Multilayer perceptron (area = {:.2f})".format(roc_auc_mlp))
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve ')
plt.legend(loc="lower right")
plt.show()

#%%%--------Phase IV: Clustering and Association----------------
#K-Means Clustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
k_values = range(2, 11)
wcss = []
silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=5085)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)
plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker='o', linestyle='-')
plt.title("Elbow Method for Optimal k using KMeans Clustering")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.xticks(k_values)
plt.grid(True)
plt.show()
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='-')
plt.title("Silhouette Analysis for Optimal k using KMeans Clustering")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.xticks(k_values)
plt.grid(True)
plt.show()

#Dbscan Algorithm
k = 57
nbrs = NearestNeighbors(n_neighbors=k).fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances[:, -1])
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title("K-Distance Graph")
plt.xlabel("Data Points (sorted)")
plt.ylabel(f"{k}-Distance")
plt.grid(True)
plt.show()
eps = 1.15
min_samples = 5
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clusters = dbscan.fit_predict(X)
df['Cluster'] = clusters
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
if n_clusters > 1:
    silhouette_avg = silhouette_score(X, clusters)
    print(f"Silhouette Score: {silhouette_avg}")
else:
    print("Silhouette Score is not defined for a single cluster.")
if isinstance(X, np.ndarray):
    X = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
plt.figure(figsize=(10, 7))
if X.shape[1] >= 2:
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='Spectral', s=20)
    plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label='Cluster Label')
    plt.show()
else:
    print("Visualization requires at least two features.")

#Apriori algorithm
df_le = df_apr[['DayOfWeek', 'Flight', 'Length', 'Time']]
df_le['DayOfWeek'] = pd.cut(df_le['DayOfWeek'], bins=3, labels=['StartOfWeek','MidOfWeek', 'Weekend'])
df_le['Flight'] = pd.cut(df_le['Flight'], bins=3, labels=['Small Aircraft', 'Medium Aircraft', 'Large Aircraft'])
df_le['Length'] = pd.cut(df_le['Length'], bins=3, labels=['Short Length', 'Average Length', 'Long Length'])
df_le['Time'] = pd.cut(df_le['Time'], bins=3, labels=['Midnight', 'Early Morning','Day'])
te = TransactionEncoder()
te_ary = te.fit(df_le.values).transform(df_le.values)
df_le = pd.DataFrame(te_ary, columns=te.columns_)
print(df_le.head(5))
df_le = df_le.astype('int')
print(df_le.head(5))
frequent_itemsets = apriori(df_le, min_support=0.1, use_colnames=True, verbose=1)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.2,num_itemsets=len(frequent_itemsets))
rules = rules.sort_values(['confidence'], ascending=False)
print(rules.head(5).to_string())




