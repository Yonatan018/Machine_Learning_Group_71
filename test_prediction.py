import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, log_loss, roc_auc_score,r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def adjust_missing_stats(df):
    drop_columns = ['PassengerId', 'Cabin']
    df.dropna(subset=drop_columns, inplace=True)

    if 'Transported' in df.columns:
        df.dropna(subset=['Transported'], inplace=True)

    mode_columns = ['CryoSleep', 'Destination', 'VIP', 'Name', 'HomePlanet']
    mean_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    for column in mode_columns:
        mode_value = df[column].mode()[0]
        df[column].fillna(mode_value, inplace=True)

    for column in mean_columns:
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)

def deduplicate(df):
    df.drop_duplicates(inplace=True)

def age_groups(df):
    df['Age_Group'] = pd.cut(df['Age'], bins=range(0, int(df['Age'].max()) + 11, 10), right=False, labels=[f'{i}-{i+9}' for i in range(0, int(df['Age'].max()) + 1, 10)])
    
    df.drop('Age',axis=1, inplace=True)

def group_size_to_categories(df):
    conditions = [
        (df['Group_Size'] == 1),
        (df['Group_Size'] >= 2) & (df['Group_Size'] <= 4),
        (df['Group_Size'] > 4)
    ]
    categories = ['1', '2-4', '>4']

    df['Group_Size_Category'] = np.select(conditions, categories, default='Unknown')
    
    df.drop('Group_Size',axis=1, inplace=True)
    
def passengerid_new_features(df):
    df["Group"] = df["PassengerId"].apply(lambda x: x.split("_")[0])
    df["Member_Number"] =df["PassengerId"].apply(lambda x: x.split("_")[1])

    x = df.groupby("Group")["Member_Number"].count().sort_values()

    set(x[x>1].index)

    df["Group_Size"]=0
    for i in x.items():
        df.loc[df["Group"] == i[0], "Group_Size"] = i[1]

    df['isInGroup'] = df['Group_Size'] > 1

    df.drop(columns=["Group", "Member_Number"], inplace=True)
    group_size_to_categories(df)

def cabin_regions(df):
    df["Cabin_Number"] = pd.to_numeric(df["Cabin_Number"], errors='coerce')
    
    max_value = df["Cabin_Number"].max()
    min_value = df["Cabin_Number"].min()
    
    range_size = (max_value - min_value) / 6
    
    region1_max = min_value + range_size
    region2_max = region1_max + range_size
    region3_max = region2_max + range_size
    region4_max = region3_max + range_size
    region5_max = region4_max + range_size
    region6_max = max_value
    
    df["Cabin_Region1"] = (df["Cabin_Number"] < region1_max)
    df["Cabin_Region2"] = (df["Cabin_Number"] >= region1_max) & (df["Cabin_Number"] < region2_max)
    df["Cabin_Region3"] = (df["Cabin_Number"] >= region2_max) & (df["Cabin_Number"] < region3_max)
    df["Cabin_Region4"] = (df["Cabin_Number"] >= region3_max) & (df["Cabin_Number"] < region4_max)
    df["Cabin_Region5"] = (df["Cabin_Number"] >= region4_max) & (df["Cabin_Number"] < region5_max)
    df["Cabin_Region6"] = (df["Cabin_Number"] >= region5_max) & (df["Cabin_Number"] <= region6_max)
    df.drop('Cabin_Number',axis=1, inplace=True)
    
def cabin_new_features(df):
    df["Cabin"].fillna("np.nan/np.nan/np.nan",inplace=True) 

    df['Cabin_Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0]).replace("np.nan",np.nan)
    df['Cabin_Number'] = df['Cabin'].apply(lambda x: x.split('/')[1]).replace("np.nan",np.nan)
    df['Cabin_Side'] = df['Cabin'].apply(lambda x: x.split('/')[2]).replace("np.nan",np.nan)
    df.drop('Cabin', axis=1, inplace=True)

    cabin_regions(df)

def expenditure_category(df):
    df["Total_Expenditure"] = pd.to_numeric(df["Total_Expenditure"], errors='coerce')
    df["Total_Expenditure_Group"] = ""
    
    df["Zero_Expenditure"] = (df["Total_Expenditure"] == 0).astype(float)
    df.loc[(df["Total_Expenditure"] > 0) & (df["Total_Expenditure"] <= 1000), "Total_Expenditure_Group"] = "Low Expense"
    df.loc[(df["Total_Expenditure"] > 1000) & (df["Total_Expenditure"] <= 2000), "Total_Expenditure_Group"] = "Medium Expense"
    df.loc[(df["Total_Expenditure"] > 2000) & (df["Total_Expenditure"] <= 4000), "Total_Expenditure_Group"] = "High Expense"
    df.loc[df["Total_Expenditure"] > 4000, "Total_Expenditure_Group"] = "Very High Expense"
    
    df.drop('Total_Expenditure',axis=1, inplace=True)

def luxury_amenities_new_feature(df):
    df['Total_Expenditure'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
    expenditure_category(df)

def df_le(df):
    label_encoder = LabelEncoder()
    columns_to_encode = ['Age_Group', 'Group_Size_Category', 'Total_Expenditure_Group']
    for col in columns_to_encode:
        df[col] = label_encoder.fit_transform(df[col])

def df_ohe(df):
    columns_to_encode = ['HomePlanet', 'Destination', 'Cabin_Deck', 'Cabin_Side']
    for col in columns_to_encode:        
        df[col] = df[col].astype('category')

    df = pd.get_dummies(df, columns=columns_to_encode, drop_first=False)
    return df

def data_to_float(df):
    to_float_columns = ['CryoSleep', 
                'HomePlanet_Europa','HomePlanet_Earth', 'HomePlanet_Mars', 'Destination_PSO J318.5-22', 
                'Destination_TRAPPIST-1e', 'Cabin_Deck_B', 'Cabin_Deck_C', 'Cabin_Deck_D', 
                'Cabin_Deck_E', 'Cabin_Deck_F', 'Cabin_Deck_G', 'Cabin_Deck_T', 'Cabin_Side_S', 
                'Destination_55 Cancri e', 'Cabin_Deck_A', 'Cabin_Side_P', 'Cabin_Region1','Cabin_Region2',
                'Cabin_Region3','Cabin_Region4','Cabin_Region5','Cabin_Region6', 'Group_Size_Category', 
                'Age_Group', 'Total_Expenditure_Group','isInGroup']
    df[to_float_columns] = df[to_float_columns].astype(float)
    if 'Transported' in df.columns:
        df['Transported'] = df['Transported'].astype(float)


def preprocess(df):
    age_groups(df)
    passengerid_new_features(df)
    cabin_new_features(df)
    luxury_amenities_new_feature(df)
    df.drop(columns =['VIP','Name'],axis=1, inplace=True)
    df_le(df)
    df = df_ohe(df)
    data_to_float(df)

    return df

# Load test data
test_df = pd.read_csv("Space_Titanic/test.csv", delimiter=',', header=0)
train_df = pd.read_csv("Space_Titanic/train.csv", delimiter=',', header=0)

# adjust for missing
adjust_missing_stats(train_df)
deduplicate(train_df)

# Preprocess test data
preprocessed_train_df = preprocess(train_df).set_index('PassengerId')
preporcessed_test_df = preprocess(test_df).set_index('PassengerId')

''' 
        Train Data: and adjust scale through PCA
'''
# Separate features and target variable
y = preprocessed_train_df['Transported']
x = preprocessed_train_df.drop(columns=['Transported'])

# Split data into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)

pca = PCA(n_components=x_train.shape[1])
x_train_pca = pca.fit_transform(x_train_scaled)
x_val_pca = pca.transform(x_val_scaled)

explained_variance = pca.explained_variance_ratio_

### 
#      Hyperparameter Tuning: using Gridsearch
###

#### Decision Tree
pipe_tree_dt = Pipeline([
    ('pca', PCA()),
    ('dt_classifier', DecisionTreeClassifier())
])

dt_param_grid = {
    'dt_classifier__criterion': ['gini', 'entropy'],
    'dt_classifier__splitter': ['best'],
    'dt_classifier__min_samples_leaf': [1, 2, 4, 6, 8, 10, 12],
    'dt_classifier__max_features': [None, 'sqrt', 'log2']
}

scoring_params = ['accuracy', 'neg_log_loss', 'roc_auc', 'f1']

# Create a GridSearchCV object
dt_grid_search = GridSearchCV(estimator=pipe_tree_dt, param_grid=dt_param_grid, cv=5, 
                              scoring=scoring_params, verbose=2, n_jobs=-1, refit='roc_auc')

# Fit the GridSearchCV object on training data
dt_grid_search.fit(x_train_scaled, y_train)

# Predict using the best estimator from grid search
grid_dt = dt_grid_search.predict(x_val_scaled)
roc_auc_dt = roc_auc_score(y_val, grid_dt)
accuracy_dt = accuracy_score(y_val, grid_dt)
f1_dt = f1_score(y_val, grid_dt)

probabilities_dt = dt_grid_search.predict_proba(x_val_scaled)
neg_log_loss_dt = -log_loss(y_val, probabilities_dt)

# Print the best estimator and best parameters
print("Best Estimator:", dt_grid_search.best_estimator_)
print("Best Parameters: ",dt_grid_search.best_params_)
print("Best Score: ", dt_grid_search.best_score_)
print("ROC AUC Score: ", roc_auc_dt)
print("Accuracy score: ", accuracy_dt)
print("F1 Score:", f1_dt)
print("Negative Log Loss:", neg_log_loss_dt)


#### kNN
knn_param_grid = {
    'knn_classifier__n_neighbors': [1,2,3,4,5,6,7,8,9,10],
    'knn_classifier__weights': ['uniform', 'distance'],
    'knn_classifier__metric': ['euclidean', 'manhattan']
}

pipe_knn = Pipeline([
    ('pca', PCA()),
    ('knn_classifier', KNeighborsClassifier())
])

knn_grid_search = GridSearchCV(estimator=pipe_knn, param_grid=knn_param_grid, cv=5,
                               scoring=scoring_params, verbose=2, n_jobs=-1, refit='roc_auc')


# Fit the GridSearchCV object on training data
knn_grid_search.fit(x_train_scaled, y_train)

# Predict using the best estimator from grid search
grid_knn = knn_grid_search.predict(x_val_scaled)
roc_auc_knn = roc_auc_score(y_val, grid_knn)
accuracy_knn = accuracy_score(y_val, grid_knn)
f1_knn = f1_score(y_val, grid_knn)

probabilities_knn = knn_grid_search.predict_proba(x_val_scaled)
neg_log_loss_knn = -log_loss(y_val, probabilities_knn)

# Print the best estimator and best parameters
print("Best Estimator:", knn_grid_search.best_estimator_)
print("Best Parameters: ",knn_grid_search.best_params_)
print("Best Score: ", knn_grid_search.best_score_)
print("ROC AUC Score: ", roc_auc_knn)
print("Accuracy score: ", accuracy_knn)
print("F1 Score:", f1_knn)
print("Negative Log Loss:", neg_log_loss_knn)

# SVC
pipe_svc = Pipeline([
    ('pca', PCA()),
    ('svc', SVC(probability=True))
])

svc_param_grid = {
    'svc__C': [0.1, 1, 10, 100, 1000],
    'svc__kernel': ['rbf'], 
    'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001]
}

svc_grid_search = GridSearchCV(estimator=pipe_svc, param_grid=svc_param_grid, cv=5, scoring=scoring_params, verbose=2, n_jobs=-1, refit='roc_auc')

svc_grid_search.fit(x_train_scaled, y_train)

# Predict using the best estimator from grid search
grid_svc = svc_grid_search.predict(x_val_scaled)
roc_auc_svc = roc_auc_score(y_val, grid_svc)
accuracy_svc = accuracy_score(y_val, grid_svc)
f1_svc = f1_score(y_val, grid_svc)

probabilities_svc = svc_grid_search.predict_proba(x_val_scaled)
neg_log_loss_svc = log_loss(y_val, probabilities_svc)
# Print the best estimator and best parameters
print("Best Estimator:",svc_grid_search.best_estimator_)
print("Best Parameters: ",svc_grid_search.best_params_)
print("Best Score: ", svc_grid_search.best_score_)
print("ROC AUC Score: ", roc_auc_svc)
print("Accuracy score: ", accuracy_svc)
print("F1 Score:", f1_svc)
print("Negative Log Loss:", neg_log_loss_svc)

### Logistic Regression
lr_param_grid = {
    'lr_classifier__C': [0.1, 1, 10],
    'lr_classifier__penalty': ['l2']
}

# Create a pipeline
pipe_lr = Pipeline([
    ('pca', PCA()),
    ('lr_classifier', LogisticRegression())
])

# Create a GridSearchCV object for Logistic Regression
lr_grid_search = GridSearchCV(estimator=pipe_lr, param_grid=lr_param_grid, cv=5, 
                              scoring=scoring_params, verbose=2, n_jobs=-1, refit='roc_auc')

# Fit GridSearchCV object on training data
lr_grid_search.fit(x_train_scaled, y_train)

# Predict using the best estimator from grid search
grid_lr = lr_grid_search.predict(x_val_scaled)
roc_auc_lr = roc_auc_score(y_val, grid_lr)
accuracy_lr = accuracy_score(y_val, grid_lr)
f1_lr = f1_score(y_val, grid_lr)

probabilities_lr = lr_grid_search.predict_proba(x_val_scaled)
neg_log_loss_lr = log_loss(y_val, probabilities_lr)

# Print the best parameters and ROC AUC score
print("Logistic Regression:")
print("Best Parameters: ", lr_grid_search.best_params_)
print("Best Score: ", lr_grid_search.best_score_)
print("ROC AUC Score: ", roc_auc_lr)
print("F1 Score:", f1_lr)
print("Negative Log Loss:", neg_log_loss_lr)

###
#       Final Predictions
###

x_test_scaled = scaler.transform(preporcessed_test_df)
x_test_pca = pca.transform(x_test_scaled)

test_predictions = svc_grid_search.predict(x_test_pca)
test_predictions_df = pd.DataFrame({'PassengerId': preporcessed_test_df.index, 'Transported': test_predictions})

test_predictions_df['Transported'] = test_predictions_df['Transported'].replace({0: False, 1: True})

test_predictions_df.to_csv('Space_Titanic/test_predictions.csv', index=False)

## Show Proportions of Transported
transported_counts = test_predictions_df['Transported'].value_counts()

proportion_true = transported_counts[True] / len(test_predictions_df)
proportion_false = transported_counts[False] / len(test_predictions_df)

print("Proportion of Transported=True:", proportion_true)
print("Proportion of Transported=False:", proportion_false)
