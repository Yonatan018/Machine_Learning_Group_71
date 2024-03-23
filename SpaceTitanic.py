# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:00.125334Z","iopub.execute_input":"2024-03-23T13:18:00.126378Z","iopub.status.idle":"2024-03-23T13:18:06.402746Z","shell.execute_reply.started":"2024-03-23T13:18:00.126334Z","shell.execute_reply":"2024-03-23T13:18:06.401526Z"}}
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import jax

from scipy import stats
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


warnings.filterwarnings("ignore")
%matplotlib inline
sns.set(style="darkgrid",font_scale=1.5)
pd.set_option("display.max.rows",None)
pd.set_option("display.max.columns",None)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [markdown]
# # Global Functions

# %% [markdown]
# # Place Transported at the end of the df

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:06.404609Z","iopub.execute_input":"2024-03-23T13:18:06.405188Z","iopub.status.idle":"2024-03-23T13:18:06.412187Z","shell.execute_reply.started":"2024-03-23T13:18:06.405154Z","shell.execute_reply":"2024-03-23T13:18:06.410792Z"}}
def transported_to_end(df):
    transported_col = df.pop('Transported')
    # Place it at the end of the DataFrame
    df['Transported'] = transported_col

# %% [markdown]
# ## Load the Data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:06.413950Z","iopub.execute_input":"2024-03-23T13:18:06.414300Z","iopub.status.idle":"2024-03-23T13:18:06.521378Z","shell.execute_reply.started":"2024-03-23T13:18:06.414257Z","shell.execute_reply":"2024-03-23T13:18:06.520260Z"}}
train_df = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv",  delimiter=',', header=0)
test_df = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv", delimiter=',', header=0)

# %% [markdown]
# ### Check size of data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:06.524897Z","iopub.execute_input":"2024-03-23T13:18:06.525224Z","iopub.status.idle":"2024-03-23T13:18:06.531243Z","shell.execute_reply.started":"2024-03-23T13:18:06.525197Z","shell.execute_reply":"2024-03-23T13:18:06.530129Z"}}
print("Training Dataset shape is: ",train_df.shape)
print("Testing Dataset shape is: ",test_df.shape)

# %% [markdown]
# # Data Preprocessing
# ### Check duplicates and remove data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:06.532922Z","iopub.execute_input":"2024-03-23T13:18:06.533394Z","iopub.status.idle":"2024-03-23T13:18:06.606341Z","shell.execute_reply.started":"2024-03-23T13:18:06.533346Z","shell.execute_reply":"2024-03-23T13:18:06.605084Z"}}
def deduplicate(df):
    duplicate_rows = df[df.duplicated()]
    df.drop_duplicates(inplace=True)
    
deduplicate(train_df)
deduplicate(test_df)

# %% [markdown]
# ## Check Missing Data
# 
# Check if we need to remove a feature by checking percentage missing values of a feature
# 
# Check if we need to remove an instance if it has too many missing data
# 
# if its a low value we later will fill na values

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:06.608104Z","iopub.execute_input":"2024-03-23T13:18:06.608528Z","iopub.status.idle":"2024-03-23T13:18:06.615332Z","shell.execute_reply.started":"2024-03-23T13:18:06.608489Z","shell.execute_reply":"2024-03-23T13:18:06.614117Z"}}
def display_missing_stats(df):
    missing_values_count = df.isnull().sum()

    # Calculate the percentage of missing values per feature
    missing_values_percentage = (missing_values_count / len(df)) * 100

    # Create a DataFrame to display the results
    missing_data_info = pd.DataFrame({
        'Missing Values': missing_values_count,
        'Percentage Missing': missing_values_percentage
    })
    print(missing_data_info)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:06.617313Z","iopub.execute_input":"2024-03-23T13:18:06.617949Z","iopub.status.idle":"2024-03-23T13:18:06.643601Z","shell.execute_reply.started":"2024-03-23T13:18:06.617907Z","shell.execute_reply":"2024-03-23T13:18:06.642421Z"}}
print('Missing Data: Training_DF\n')
display_missing_stats(train_df)

# %% [markdown]
# ## Remove Missing Data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:06.645213Z","iopub.execute_input":"2024-03-23T13:18:06.645544Z","iopub.status.idle":"2024-03-23T13:18:06.671037Z","shell.execute_reply.started":"2024-03-23T13:18:06.645516Z","shell.execute_reply":"2024-03-23T13:18:06.669983Z"}}
drop_columns = ['PassengerId', 'Cabin']
train_df.dropna(subset=drop_columns, inplace=True)
display_missing_stats(train_df)
print("Training Dataset shape is: ",train_df.shape)

# %% [markdown]
# ## Fill Missing

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:06.672357Z","iopub.execute_input":"2024-03-23T13:18:06.672710Z","iopub.status.idle":"2024-03-23T13:18:06.682653Z","shell.execute_reply.started":"2024-03-23T13:18:06.672667Z","shell.execute_reply":"2024-03-23T13:18:06.681423Z"}}
train_df.dtypes

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:06.689309Z","iopub.execute_input":"2024-03-23T13:18:06.689675Z","iopub.status.idle":"2024-03-23T13:18:06.732448Z","shell.execute_reply.started":"2024-03-23T13:18:06.689645Z","shell.execute_reply":"2024-03-23T13:18:06.731095Z"}}
mode_columns = ['CryoSleep', 'Destination', 'VIP', 'Name', 'HomePlanet']
mean_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Fill missing values with mode for mode_columns
for column in mode_columns:
    mode_value = train_df[column].mode()[0]
    train_df[column].fillna(mode_value, inplace=True)

# Fill missing values with mean for mean_columns
for column in mean_columns:
    mean_value = train_df[column].mean()
    train_df[column].fillna(mean_value, inplace=True)


display_missing_stats(train_df)
print("Training Dataset shape is: ",train_df.shape)

# %% [markdown]
# ### Change boolean features from string to int

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:06.733883Z","iopub.execute_input":"2024-03-23T13:18:06.735180Z","iopub.status.idle":"2024-03-23T13:18:06.775347Z","shell.execute_reply.started":"2024-03-23T13:18:06.735141Z","shell.execute_reply":"2024-03-23T13:18:06.774299Z"}}
train_df[['CryoSleep', 'VIP', 'Transported']] = train_df[['CryoSleep', 'VIP', 'Transported']].replace({True: 1, False: 0})
test_df[['CryoSleep', 'VIP']] = test_df[['CryoSleep', 'VIP']].replace({True: 1, False: 0})

# %% [markdown]
# ## Check Outliers

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:06.776682Z","iopub.execute_input":"2024-03-23T13:18:06.777226Z","iopub.status.idle":"2024-03-23T13:18:06.784997Z","shell.execute_reply.started":"2024-03-23T13:18:06.777195Z","shell.execute_reply":"2024-03-23T13:18:06.783778Z"}}
def get_df_name(df):
    # Use globals() to obtain the variable name as a string
    for name, var in globals().items():
        if var is df:
            return name
    return None  # Return None if the DataFrame is not found

# Selecting only numerical columns for outlier detection
def show_outliers(df):
    numerical_columns = df.select_dtypes(include=['float64']).columns

    # Display summary statistics for numerical columns
    summary_stats = df[numerical_columns].describe()

    # Display box plots for numerical columns
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df[numerical_columns])
    plt.title(f'Box plots of Numerical Columns {get_df_name(df)}')

    plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:06.786435Z","iopub.execute_input":"2024-03-23T13:18:06.787446Z","iopub.status.idle":"2024-03-23T13:18:07.808816Z","shell.execute_reply.started":"2024-03-23T13:18:06.787404Z","shell.execute_reply":"2024-03-23T13:18:07.807719Z"}}
show_outliers(train_df)
show_outliers(test_df)

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# ## New Features from PassengerID
# PassengerId -> gggg_pp, 
# - g -> group passenger is travelling with
# - p -> # people within group
# create new feature for group size
# create a new feature for solo traveling

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:07.810343Z","iopub.execute_input":"2024-03-23T13:18:07.811003Z","iopub.status.idle":"2024-03-23T13:18:07.820455Z","shell.execute_reply.started":"2024-03-23T13:18:07.810963Z","shell.execute_reply":"2024-03-23T13:18:07.819342Z"}}
def passengerid_new_features(df):
    #Splitting Group and Member values from "PassengerId" column.
    df["Group"] = df["PassengerId"].apply(lambda x: x.split("_")[0])
    df["Member_Number"] =df["PassengerId"].apply(lambda x: x.split("_")[1])

    #Grouping the "Group" feature with respect to "member" feature to check which group is travelling with how many members
    x = df.groupby("Group")["Member_Number"].count().sort_values()

    #Creating a set of group values which are travelling with more than 1 members.
    y = set(x[x>1].index)

    #Creating a new feature "Group_size" which will indicate each group number of members.
    df["Group_Size"]=0
    for i in x.items():
        df.loc[df["Group"] == i[0], "Group_Size"] = i[1]

    # Adding isInGroup feature
    df['isInGroup'] = df['Group_Size'] > 1

    # Dropping unnecessary columns
    df.drop(columns=["PassengerId", "Group", "Member_Number"], inplace=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:07.821728Z","iopub.execute_input":"2024-03-23T13:18:07.822093Z","iopub.status.idle":"2024-03-23T13:18:24.375533Z","shell.execute_reply.started":"2024-03-23T13:18:07.822058Z","shell.execute_reply":"2024-03-23T13:18:24.374457Z"}}
passengerid_new_features(train_df)
passengerid_new_features(test_df)

# %% [markdown]
# ## New Features from Cabin
# Create new Features from Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:24.376959Z","iopub.execute_input":"2024-03-23T13:18:24.377269Z","iopub.status.idle":"2024-03-23T13:18:24.385296Z","shell.execute_reply.started":"2024-03-23T13:18:24.377242Z","shell.execute_reply":"2024-03-23T13:18:24.383986Z"}}
def cabin_new_features(df):
    df["Cabin"].fillna("np.nan/np.nan/np.nan",inplace=True) 

    # Create new features: Deck, CabinNumber, Side
    df['Cabin_Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0]).replace("np.nan",np.nan)
    df['Cabin_Number'] = df['Cabin'].apply(lambda x: x.split('/')[1]).replace("np.nan",np.nan)
    df['Cabin_Side'] = df['Cabin'].apply(lambda x: x.split('/')[2]).replace("np.nan",np.nan)

    df.drop('Cabin',axis=1, inplace=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:24.386961Z","iopub.execute_input":"2024-03-23T13:18:24.387436Z","iopub.status.idle":"2024-03-23T13:18:24.446205Z","shell.execute_reply.started":"2024-03-23T13:18:24.387392Z","shell.execute_reply":"2024-03-23T13:18:24.445131Z"}}
cabin_new_features(train_df)
cabin_new_features(test_df)

# %% [markdown]
# # Total Expenditure
# RoomService, FoodCourt, ShoppingMall, Spa, VRDeck summed up

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:24.447278Z","iopub.execute_input":"2024-03-23T13:18:24.447572Z","iopub.status.idle":"2024-03-23T13:18:24.453210Z","shell.execute_reply.started":"2024-03-23T13:18:24.447546Z","shell.execute_reply":"2024-03-23T13:18:24.452052Z"}}
def luxury_amenities_new_feature(df):
    df['Total_Expenditure'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:24.454892Z","iopub.execute_input":"2024-03-23T13:18:24.455338Z","iopub.status.idle":"2024-03-23T13:18:24.473366Z","shell.execute_reply.started":"2024-03-23T13:18:24.455271Z","shell.execute_reply":"2024-03-23T13:18:24.472419Z"}}
luxury_amenities_new_feature(train_df)
luxury_amenities_new_feature(test_df)

# %% [markdown]
# # Graphs and Figures

# %% [markdown]
# ## Transported Distribution

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:24.474301Z","iopub.execute_input":"2024-03-23T13:18:24.474601Z","iopub.status.idle":"2024-03-23T13:18:24.821931Z","shell.execute_reply.started":"2024-03-23T13:18:24.474575Z","shell.execute_reply":"2024-03-23T13:18:24.820025Z"}}
plt.figure(figsize=(10,6))
plt.pie(train_df["Transported"].value_counts(),labels=train_df["Transported"].value_counts().keys(),autopct="%1.1f%%",
       textprops={"fontsize":20,"fontweight":"black"},colors=sns.color_palette("Set2"))
plt.title("Transported Feature Distribution");

# %% [markdown]
# ## Current Dataframe After Processing

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:24.824684Z","iopub.execute_input":"2024-03-23T13:18:24.825300Z","iopub.status.idle":"2024-03-23T13:18:24.839379Z","shell.execute_reply.started":"2024-03-23T13:18:24.825249Z","shell.execute_reply":"2024-03-23T13:18:24.837082Z"}}
def set_teleported_last(df):
    if 'Transported' in df.columns:
        transported_col = df.pop('Transported')
        df['Transported'] = transported_col

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:24.842102Z","iopub.execute_input":"2024-03-23T13:18:24.842755Z","iopub.status.idle":"2024-03-23T13:18:24.879583Z","shell.execute_reply.started":"2024-03-23T13:18:24.842689Z","shell.execute_reply":"2024-03-23T13:18:24.878317Z"}}
set_teleported_last(train_df)
set_teleported_last(test_df)

train_df.head()

# %% [markdown]
# ### Compare HomePlanet with Transported

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:24.881146Z","iopub.execute_input":"2024-03-23T13:18:24.881466Z","iopub.status.idle":"2024-03-23T13:18:25.230229Z","shell.execute_reply.started":"2024-03-23T13:18:24.881439Z","shell.execute_reply":"2024-03-23T13:18:25.228918Z"}}
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(x='HomePlanet', hue='Transported', data=train_df, palette="Set2")
plt.title("HomePlanet Distribution")
plt.show()

# %% [markdown]
# ### Compare CryoSleep with Transported

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:25.232042Z","iopub.execute_input":"2024-03-23T13:18:25.232481Z","iopub.status.idle":"2024-03-23T13:18:25.568002Z","shell.execute_reply.started":"2024-03-23T13:18:25.232432Z","shell.execute_reply":"2024-03-23T13:18:25.566905Z"}}
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(x='CryoSleep', hue='Transported', data=train_df, palette="Set2")
plt.title("CryoSleep Distribution")
plt.show()

# %% [markdown]
# ### Compare Destination with Transported

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:25.569447Z","iopub.execute_input":"2024-03-23T13:18:25.570162Z","iopub.status.idle":"2024-03-23T13:18:25.922562Z","shell.execute_reply.started":"2024-03-23T13:18:25.570123Z","shell.execute_reply":"2024-03-23T13:18:25.921445Z"}}
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(x='Destination', hue='Transported', data=train_df, palette="Set2")
plt.title("Destination Distribution")
plt.show()

# %% [markdown]
# ### Compare Age with Transported

# %% [markdown]
# Show age distribution without grouping

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:25.923950Z","iopub.execute_input":"2024-03-23T13:18:25.924386Z","iopub.status.idle":"2024-03-23T13:18:26.910288Z","shell.execute_reply.started":"2024-03-23T13:18:25.924345Z","shell.execute_reply":"2024-03-23T13:18:26.909115Z"}}
plt.figure(figsize=(16,6))
sns.histplot(x=train_df["Age"],hue="Transported",data=train_df,kde=True,palette="Set2")
plt.title("Age Feature Distribution");

# %% [markdown]
# Find the Age range, then create age groups of 5

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:26.911958Z","iopub.execute_input":"2024-03-23T13:18:26.912463Z","iopub.status.idle":"2024-03-23T13:18:26.930480Z","shell.execute_reply.started":"2024-03-23T13:18:26.912424Z","shell.execute_reply":"2024-03-23T13:18:26.929136Z"}}
age_range = train_df['Age'].describe()[['min', 'max']]
print(age_range)
train_df['Age_Group'] = pd.cut(train_df['Age'], bins=range(0, int(train_df['Age'].max()) + 11, 10), right=False, labels=[f'{i}-{i+9}' for i in range(0, int(train_df['Age'].max()) + 1, 10)])

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:26.939023Z","iopub.execute_input":"2024-03-23T13:18:26.939372Z","iopub.status.idle":"2024-03-23T13:18:27.380449Z","shell.execute_reply.started":"2024-03-23T13:18:26.939345Z","shell.execute_reply":"2024-03-23T13:18:27.379277Z"}}
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(x='Age_Group', hue='Transported', data=train_df, palette="Set2")
plt.title("Age Distribution")
plt.show()

# %% [markdown]
# ### Compare VIP with Transported

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:27.381759Z","iopub.execute_input":"2024-03-23T13:18:27.382172Z","iopub.status.idle":"2024-03-23T13:18:27.721162Z","shell.execute_reply.started":"2024-03-23T13:18:27.382141Z","shell.execute_reply":"2024-03-23T13:18:27.720021Z"}}
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(x='VIP', hue='Transported', data=train_df, palette="Set2")
plt.title("VIP Distribution")
plt.show()

# %% [markdown]
# Show proportion of diffence

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:27.722725Z","iopub.execute_input":"2024-03-23T13:18:27.723822Z","iopub.status.idle":"2024-03-23T13:18:27.732419Z","shell.execute_reply.started":"2024-03-23T13:18:27.723778Z","shell.execute_reply":"2024-03-23T13:18:27.731081Z"}}
target_count = train_df['VIP'].value_counts()
print('VIP:', target_count[1])
print('Non_VIP:', target_count[0])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

# %% [markdown]
# ### Compare Group_Size with Transported

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:27.734150Z","iopub.execute_input":"2024-03-23T13:18:27.734639Z","iopub.status.idle":"2024-03-23T13:18:28.161991Z","shell.execute_reply.started":"2024-03-23T13:18:27.734571Z","shell.execute_reply":"2024-03-23T13:18:28.160779Z"}}
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(x='Group_Size', hue='Transported', data=train_df, palette="Set2")
plt.title("Group_Size Distribution")
plt.show()

# %% [markdown]
# ### Compare Deck with Transported

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:28.163441Z","iopub.execute_input":"2024-03-23T13:18:28.163809Z","iopub.status.idle":"2024-03-23T13:18:28.625344Z","shell.execute_reply.started":"2024-03-23T13:18:28.163778Z","shell.execute_reply":"2024-03-23T13:18:28.624219Z"}}
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(x='Cabin_Deck', hue='Transported', data=train_df, palette="Set2", order=['A', 'B', 'C', 'D', 'E','F','G','T'])
plt.title("Deck Distribution")
plt.show()

# %% [markdown]
# ### Compare Cabin_Number with Transported

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:28.626920Z","iopub.execute_input":"2024-03-23T13:18:28.629063Z","iopub.status.idle":"2024-03-23T13:18:40.458618Z","shell.execute_reply.started":"2024-03-23T13:18:28.629018Z","shell.execute_reply":"2024-03-23T13:18:40.457499Z"}}
plt.figure(figsize=(15, 5))
subset_df = train_df.sample(frac=1)  # Adjust the fraction as needed
sns.histplot(x="Cabin_Number", data=train_df, hue="Transported", palette="Set2")
plt.title("Cabin_Number Distribution")
plt.xticks(list(range(0, 1900, 300)))
plt.vlines(300, ymin=0, ymax=15, color="black")
plt.vlines(600, ymin=0, ymax=15, color="black")
plt.vlines(900, ymin=0, ymax=15, color="black")
plt.vlines(1200, ymin=0, ymax=15, color="black")
plt.vlines(1500, ymin=0, ymax=15, color="black")
plt.show()

# %% [markdown]
# Create Cabin Categories

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:40.460098Z","iopub.execute_input":"2024-03-23T13:18:40.460444Z","iopub.status.idle":"2024-03-23T13:18:40.468365Z","shell.execute_reply.started":"2024-03-23T13:18:40.460415Z","shell.execute_reply":"2024-03-23T13:18:40.467120Z"}}
def cabin_regions(df):
    df["Cabin_Number"] = pd.to_numeric(df["Cabin_Number"], errors='coerce')  # Convert to numeric type
    df["Cabin_Region1"] = (df["Cabin_Number"]<252)
    df["Cabin_Region2"] = (df["Cabin_Number"]>=252) & (df["Cabin_Number"]<638)
    df["Cabin_Region3"] = (df["Cabin_Number"]>=638) & (df["Cabin_Number"]<870)
    df["Cabin_Region4"] = (df["Cabin_Number"]>=870) & (df["Cabin_Number"]<1266)
    df["Cabin_Region5"] = (df["Cabin_Number"]>=1266) & (df["Cabin_Number"]<1597)
    df["Cabin_Region6"] = (df["Cabin_Number"]>=1876)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:40.469457Z","iopub.execute_input":"2024-03-23T13:18:40.469802Z","iopub.status.idle":"2024-03-23T13:18:40.499968Z","shell.execute_reply.started":"2024-03-23T13:18:40.469769Z","shell.execute_reply":"2024-03-23T13:18:40.499054Z"}}
cabin_regions(train_df)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:40.501238Z","iopub.execute_input":"2024-03-23T13:18:40.502051Z","iopub.status.idle":"2024-03-23T13:18:43.314008Z","shell.execute_reply.started":"2024-03-23T13:18:40.502010Z","shell.execute_reply":"2024-03-23T13:18:43.312805Z"}}
cols = ["Cabin_Region1","Cabin_Region2","Cabin_Region3","Cabin_Region4","Cabin_Region5","Cabin_Region6"]

plt.figure(figsize=(20,25))
for idx,value in enumerate(cols):
    plt.subplot(4,2,idx+1)
    sns.countplot(x=value, hue="Transported", data=train_df, palette="Set2")
    plt.title(f"{value} Distribution")
    plt.tight_layout()

# %% [markdown]
# ### Check if people in the same cabin are transported
# 
# The data shows that if someone gets transported, then at least 30% of the people sharing the same room disappear.
# It is either a majority disappears or a majority doesn't disappear, which may show correlation between Cabin Number and Transported.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:43.315605Z","iopub.execute_input":"2024-03-23T13:18:43.315975Z","iopub.status.idle":"2024-03-23T13:18:44.093759Z","shell.execute_reply.started":"2024-03-23T13:18:43.315945Z","shell.execute_reply":"2024-03-23T13:18:44.092653Z"}}
# Set the style of seaborn
sns.set(style="whitegrid")

# Filter rows where 'Transported' is 1
transported_1_df = train_df[train_df['Transported'] == 1]

# Group by 'Cabin_Number' and 'Group_Size' and calculate the count
cabin_group_counts_1 = transported_1_df.groupby(['Cabin_Number', 'Group_Size']).size().unstack(fill_value=0)

# Calculate the proportion for each 'Group_Size' when 'Transported' is 1
cabin_group_proportion_1 = cabin_group_counts_1.div(cabin_group_counts_1.sum(axis=1), axis=0)

# Create groups for proportions with intervals of 10
cabin_group_proportion_1['Proportion_Group'] = pd.cut(cabin_group_proportion_1.max(axis=1) * 100, bins=range(0, 110, 10), right=False)

# Count the occurrences of each Proportion_Group for transported 1
group_counts_1 = cabin_group_proportion_1['Proportion_Group'].value_counts(sort=False)

# Filter rows where 'Transported' is 0
transported_0_df = train_df[train_df['Transported'] == 0]

# Group by 'Cabin_Number' and 'Group_Size' and calculate the count
cabin_group_counts_0 = transported_0_df.groupby(['Cabin_Number', 'Group_Size']).size().unstack(fill_value=0)

# Calculate the proportion for each 'Group_Size' when 'Transported' is 0
cabin_group_proportion_0 = cabin_group_counts_0.div(cabin_group_counts_0.sum(axis=1), axis=0)

# Create groups for proportions with intervals of 10
cabin_group_proportion_0['Proportion_Group'] = pd.cut(cabin_group_proportion_0.max(axis=1) * 100, bins=range(0, 110, 10), right=False)

# Count the occurrences of each Proportion_Group for transported 0
group_counts_0 = cabin_group_proportion_0['Proportion_Group'].value_counts(sort=False)

# Plot the proportions for transported 1 and transported 0 side by side
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# Plot for transported 1
sns.barplot(ax=axes[0], x=group_counts_1.index.astype(str), y=group_counts_1.values, color='skyblue')
axes[0].set_title("Counts of Cabin Numbers within Proportion Groups (Transported=1)")
axes[0].set_xlabel("Proportion Groups (in percentage)")
axes[0].set_ylabel("Count")

# Plot for transported 0
sns.barplot(ax=axes[1], x=group_counts_0.index.astype(str), y=group_counts_0.values, color='lightgreen')
axes[1].set_title("Counts of Cabin Numbers within Proportion Groups (Transported=0)")
axes[1].set_xlabel("Proportion Groups (in percentage)")
axes[1].set_ylabel("Count")

plt.show()

# %% [markdown]
# ### Compare Side with Transported

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:44.095134Z","iopub.execute_input":"2024-03-23T13:18:44.095472Z","iopub.status.idle":"2024-03-23T13:18:44.413530Z","shell.execute_reply.started":"2024-03-23T13:18:44.095443Z","shell.execute_reply":"2024-03-23T13:18:44.412541Z"}}
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(x='Cabin_Side', hue='Transported', data=train_df, palette="Set2")
plt.title("Side Distribution")
plt.show()

# %% [markdown]
# ### Compare Total_Expenditure with Transported

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:44.414688Z","iopub.execute_input":"2024-03-23T13:18:44.415036Z","iopub.status.idle":"2024-03-23T13:18:45.812641Z","shell.execute_reply.started":"2024-03-23T13:18:44.415008Z","shell.execute_reply":"2024-03-23T13:18:45.811573Z"}}
plt.figure(figsize=(15, 6))
sns.histplot(x='Total_Expenditure', hue='Transported', data=train_df, kde=True, palette='Set2', bins=200)
plt.title("Total_Expenditure Distribution")
plt.ylim(0, 1000)
plt.xlim(0, 10000)
plt.show()

# %% [markdown]
# ### split group_size into categories

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:45.814453Z","iopub.execute_input":"2024-03-23T13:18:45.814816Z","iopub.status.idle":"2024-03-23T13:18:45.824722Z","shell.execute_reply.started":"2024-03-23T13:18:45.814787Z","shell.execute_reply":"2024-03-23T13:18:45.823532Z"}}
# Define the conditions for categorizing Group_Size
conditions = [
    (train_df['Group_Size'] == 1),
    (train_df['Group_Size'] >= 2) & (train_df['Group_Size'] <= 4),
    (train_df['Group_Size'] > 4)
]

# Define the category labels
categories = ['1', '2-4', '>4']

# Create a new column 'Group_Size_Category' based on the conditions
train_df['Group_Size_Category'] = np.select(conditions, categories, default='Unknown')

# %% [markdown]
# ### Make 0 spending a separate feature
# 
# Take out people with 0 expenditure and make it a separate feature
# Replot expenditure to find the different categories of Expenditure

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:45.826163Z","iopub.execute_input":"2024-03-23T13:18:45.826566Z","iopub.status.idle":"2024-03-23T13:18:46.128875Z","shell.execute_reply.started":"2024-03-23T13:18:45.826535Z","shell.execute_reply":"2024-03-23T13:18:46.127983Z"}}
train_df['Zero_Expenditure'] = train_df['Total_Expenditure'].apply(lambda x: 1 if x == 0 else 0)
# Set the style of seaborn
sns.set(style="whitegrid")

# Filter rows where 'Transported' is 0 and 1 separately
transported_0_zero_exp_df = train_df[(train_df['Transported'] == 0) & (train_df['Zero_Expenditure'] == 1)]
transported_1_zero_exp_df = train_df[(train_df['Transported'] == 1) & (train_df['Zero_Expenditure'] == 1)]

# Count the occurrences of zero expenditure for 'Transported' values of 0 and 1
zero_exp_counts = pd.DataFrame({
    'Transported_0': [len(transported_0_zero_exp_df)],
    'Transported_1': [len(transported_1_zero_exp_df)]
})

# Plot the counts
plt.figure(figsize=(8, 6))
sns.barplot(data=zero_exp_counts, palette="Set2")
plt.title("Count of Zero Expenditure by Transported Status")
plt.xlabel("Transported")
plt.ylabel("Count of Zero Expenditure")
plt.xticks(rotation=0)
plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:46.130322Z","iopub.execute_input":"2024-03-23T13:18:46.130934Z","iopub.status.idle":"2024-03-23T13:18:46.164371Z","shell.execute_reply.started":"2024-03-23T13:18:46.130903Z","shell.execute_reply":"2024-03-23T13:18:46.163358Z"}}
transported_to_end(train_df)
train_df.head()

# %% [markdown]
# # Model Building

# %% [markdown]
# ## Create Groups for Total_Expenditure

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:46.165633Z","iopub.execute_input":"2024-03-23T13:18:46.166145Z","iopub.status.idle":"2024-03-23T13:18:46.173534Z","shell.execute_reply.started":"2024-03-23T13:18:46.166114Z","shell.execute_reply":"2024-03-23T13:18:46.172596Z"}}
def expenditure_category(df):
    df["Total_Expenditure"] = pd.to_numeric(df["Total_Expenditure"], errors='coerce')
    df["Total_Expenditure_Group"] = ""
    df.loc[df["Total_Expenditure"] == 0, "Total_Expenditure_Group"] = "No Expense"
    df.loc[(df["Total_Expenditure"] > 0) & (df["Total_Expenditure"] <= 1000), "Total_Expenditure_Group"] = "Low Expense"
    df.loc[(df["Total_Expenditure"] > 1000) & (df["Total_Expenditure"] <= 2000), "Total_Expenditure_Group"] = "Medium Expense"
    df.loc[(df["Total_Expenditure"] > 2000) & (df["Total_Expenditure"] <= 4000), "Total_Expenditure_Group"] = "High Expense"
    df.loc[df["Total_Expenditure"] > 4000, "Total_Expenditure_Group"] = "Very High Expense"

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:46.175264Z","iopub.execute_input":"2024-03-23T13:18:46.175678Z","iopub.status.idle":"2024-03-23T13:18:46.197765Z","shell.execute_reply.started":"2024-03-23T13:18:46.175639Z","shell.execute_reply":"2024-03-23T13:18:46.196427Z"}}
expenditure_category(train_df)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:46.199297Z","iopub.execute_input":"2024-03-23T13:18:46.199744Z","iopub.status.idle":"2024-03-23T13:18:46.238193Z","shell.execute_reply.started":"2024-03-23T13:18:46.199708Z","shell.execute_reply":"2024-03-23T13:18:46.236957Z"}}
train_df.drop(columns = ['Name', 'VIP', 'Age', 'Total_Expenditure', 'Group_Size', 'Cabin_Number'],axis = 1, inplace =True)
train_df.head()

# %% [markdown]
# # Category to Numeric

# %% [markdown]
# ## Label Encoding

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:46.239772Z","iopub.execute_input":"2024-03-23T13:18:46.240143Z","iopub.status.idle":"2024-03-23T13:18:46.256884Z","shell.execute_reply.started":"2024-03-23T13:18:46.240113Z","shell.execute_reply":"2024-03-23T13:18:46.255746Z"}}
def df_le(df):
    label_encoder = LabelEncoder()
    columns_to_encode = ['Age_Group', 'Group_Size_Category', 'Total_Expenditure_Group']
    for col in columns_to_encode:
        # Fit label encoder and transform values to numerical labels
        df[col] = label_encoder.fit_transform(df[col])

df_le(train_df)

# %% [markdown]
# ## One Hot Encoding:

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:46.258528Z","iopub.execute_input":"2024-03-23T13:18:46.258845Z","iopub.status.idle":"2024-03-23T13:18:46.309186Z","shell.execute_reply.started":"2024-03-23T13:18:46.258819Z","shell.execute_reply":"2024-03-23T13:18:46.308065Z"}}
def df_ohe(df):
    columns_to_encode = ['HomePlanet', 'Destination', 'Cabin_Deck', 'Cabin_Side']
    for col in columns_to_encode:        
        df[col] = df[col].astype('category')

    df = pd.get_dummies(df, columns=columns_to_encode, drop_first=False)
    
    return df

train_df = df_ohe(train_df)
train_df.head()

# %% [markdown]
# ## Change Category to Numeric

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:46.311501Z","iopub.execute_input":"2024-03-23T13:18:46.311971Z","iopub.status.idle":"2024-03-23T13:18:46.374444Z","shell.execute_reply.started":"2024-03-23T13:18:46.311929Z","shell.execute_reply":"2024-03-23T13:18:46.373333Z"}}
bool_columns = ['CryoSleep', 
                'HomePlanet_Europa','HomePlanet_Earth', 'HomePlanet_Mars', 'Destination_PSO J318.5-22', 
                'Destination_TRAPPIST-1e', 'Cabin_Deck_B', 'Cabin_Deck_C', 'Cabin_Deck_D', 
                'Cabin_Deck_E', 'Cabin_Deck_F', 'Cabin_Deck_G', 'Cabin_Deck_T', 'Cabin_Side_S', 
                'Destination_55 Cancri e', 'Cabin_Deck_A', 'Cabin_Side_P', 'Cabin_Region1','Cabin_Region2',
                'Cabin_Region3','Cabin_Region4','Cabin_Region5','Cabin_Region6', 'Group_Size_Category', 
                'Age_Group', 'Transported', 'Total_Expenditure_Group','isInGroup']

# Convert boolean columns to float64
train_df[bool_columns] = train_df[bool_columns].astype(float)
final_train = train_df
# Display the modified DataFrame with object types
print(final_train.dtypes)
final_train.head()

# %% [markdown]
# # Find significant features:

# %% [markdown]
# ### Use Correlation Matrix

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:46.375773Z","iopub.execute_input":"2024-03-23T13:18:46.376100Z","iopub.status.idle":"2024-03-23T13:18:50.908887Z","shell.execute_reply.started":"2024-03-23T13:18:46.376071Z","shell.execute_reply":"2024-03-23T13:18:50.907884Z"}}
# Calculate correlation matrix
correlation_matrix = final_train.corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(30, 20))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:50.910082Z","iopub.execute_input":"2024-03-23T13:18:50.911449Z","iopub.status.idle":"2024-03-23T13:18:50.946981Z","shell.execute_reply.started":"2024-03-23T13:18:50.911415Z","shell.execute_reply":"2024-03-23T13:18:50.945292Z"}}
# Filter out correlations equal to 1
corr_matrix_filtered = correlation_matrix[correlation_matrix != 1]

# Flatten the correlation matrix and sort the values
flattened_corr = corr_matrix_filtered.unstack().sort_values(ascending=False)

# Remove duplicates by filtering out correlations where the index is greater than the column
# This ensures we only keep one direction of the correlation
unique_corr = flattened_corr[flattened_corr.index.get_level_values(0) < flattened_corr.index.get_level_values(1)]

# %% [markdown]
# ## Training

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:50.949024Z","iopub.execute_input":"2024-03-23T13:18:50.949614Z","iopub.status.idle":"2024-03-23T13:18:50.972394Z","shell.execute_reply.started":"2024-03-23T13:18:50.949573Z","shell.execute_reply":"2024-03-23T13:18:50.971252Z"}}
y = final_train['Transported']
x = final_train.drop(columns=['Transported'])

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle = True, stratify=y)

# %% [markdown]
# ## Recursive Feature Eliminator (RFE)
# ### Logistic Regression

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:50.974435Z","iopub.execute_input":"2024-03-23T13:18:50.974800Z","iopub.status.idle":"2024-03-23T13:18:50.980548Z","shell.execute_reply.started":"2024-03-23T13:18:50.974769Z","shell.execute_reply":"2024-03-23T13:18:50.979415Z"}}
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'kNN': KNeighborsClassifier(),
    'SVC': SVC(),
    'Linear Regression': GradientBoostingClassifier(),
    'Gausian Classifier': GaussianNB()
}

# %% [markdown]
# # Hyperparameter Tuning

# %% [markdown]
# ### Decision Tree

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:56.263221Z","iopub.execute_input":"2024-03-23T13:18:56.263742Z","iopub.status.idle":"2024-03-23T13:19:20.851536Z","shell.execute_reply.started":"2024-03-23T13:18:56.263692Z","shell.execute_reply":"2024-03-23T13:19:20.850484Z"}}
# Define parameters grid for Decision Tree
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Initialize Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Create a GridSearchCV object
dt_grid_search = GridSearchCV(estimator=dt_classifier, param_grid=dt_param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit the GridSearchCV object on training data
dt_grid_search.fit(x_train, y_train)

# Print the best parameters found by the grid search
print("Best Parameters for Decision Tree:", dt_grid_search.best_params_)

# Get the best Decision Tree classifier model
best_dt_model = dt_grid_search.best_estimator_

# Evaluate the best Decision Tree model on the validation set
y_pred_dt = best_dt_model.predict(x_val[selected_features])
accuracy_dt = accuracy_score(y_val, y_pred_dt)
print("Accuracy of Best Decision Tree Model:", accuracy_dt)

# %% [markdown]
# ### kNN Classifier

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:19:20.855215Z","iopub.execute_input":"2024-03-23T13:19:20.855528Z","iopub.status.idle":"2024-03-23T13:19:37.773342Z","shell.execute_reply.started":"2024-03-23T13:19:20.855501Z","shell.execute_reply":"2024-03-23T13:19:37.772257Z"}}
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Initialize kNN Classifier
knn_classifier = KNeighborsClassifier()

# Create a GridSearchCV object
knn_grid_search = GridSearchCV(estimator=knn_classifier, param_grid=knn_param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit the GridSearchCV object on training data
knn_grid_search.fit(x_train, y_train)

# Print the best parameters found by the grid search
print("Best Parameters for kNN:", knn_grid_search.best_params_)

# Get the best kNN classifier model
best_knn_model = knn_grid_search.best_estimator_

# Evaluate the best kNN model on the validation set
y_pred_knn = best_knn_model.predict(x_val)
accuracy_knn = accuracy_score(y_val, y_pred_knn)
print("Accuracy of Best kNN Model:", accuracy_knn)

# %% [markdown]
# ### Support Vector Matrix
# 
# svc kernel -> rbf kernel (not linear)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:19:37.774994Z","iopub.execute_input":"2024-03-23T13:19:37.775315Z"}}
svc_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Initialize SVC Classifier
svc_classifier = SVC(random_state=42)

# Create a GridSearchCV object
svc_grid_search = GridSearchCV(estimator=svc_classifier, param_grid=svc_param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit the GridSearchCV object on training data
svc_grid_search.fit(x_train, y_train)

# Print the best parameters found by the grid search
print("Best Parameters for SVC:", svc_grid_search.best_params_)

# Get the best SVC classifier model
best_svc_model = svc_grid_search.best_estimator_

# Evaluate the best SVC model on the validation set
y_pred_svc = best_svc_model.predict(x_val)
accuracy_svc = accuracy_score(y_val, y_pred_svc)
print("Accuracy of Best SVC Model:", accuracy_svc)

# %% [markdown]
# ### Gautian

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Initialize Gaussian Naive Bayes Classifier
gnb_classifier = GaussianNB()

# Fit the classifier on training data
gnb_classifier.fit(x_train, y_train)

# Predict on validation data
y_pred_gnb = gnb_classifier.predict(x_val)

# Calculate accuracy
accuracy_gnb = accuracy_score(y_val, y_pred_gnb)
print("Accuracy of Gaussian Naive Bayes Model:", accuracy_gnb)

# %% [markdown]
# ### linear Regression

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Initialize Logistic Regression Classifier
lr_classifier = LogisticRegression(random_state=42)

# Fit the classifier on training data
lr_classifier.fit(x_train, y_train)

# Predict on validation data
y_pred_lr = lr_classifier.predict(x_val)

# Calculate accuracy
accuracy_lr = accuracy_score(y_val, y_pred_lr)
print("Accuracy of Logistic Regression Model:", accuracy_lr)

# %% [markdown]
# ## Plot Decision Tree

# %% [code] {"jupyter":{"outputs_hidden":false}}
# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(x_train[selected_features], y_train)
# plt.figure(figsize=(20, 10))
# plot_tree(decision_tree, filled=True, feature_names=selected_features, class_names=['Class 0', 'Class 1'])
# plt.show()

# %% [markdown]
# ### Evaluate Models

# %% [code]
# Train and evaluate each model
for model_name, model in models.items():
    # Train model on selected features
    model.fit(x_train[selected_features], y_train)
    
    # Predict on validation data
    y_pred = model.predict(x_val[x_train])
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    
    # Print accuracy for each model
    print(f'{model_name} Accuracy: {accuracy}')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-23T13:18:56.250984Z","iopub.execute_input":"2024-03-23T13:18:56.251486Z","iopub.status.idle":"2024-03-23T13:18:56.261872Z","shell.execute_reply.started":"2024-03-23T13:18:56.251456Z","shell.execute_reply":"2024-03-23T13:18:56.260788Z"}}
# Define a function to plot ROC curve and Precision-Recall curve
def plot_curves(model, x_val, y_val, model_name):
    if hasattr(model, "predict_proba"):
        # Predict probabilities
        y_probs = model.predict_proba(x_val[selected_features])[:, 1]

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_val, y_probs)
        roc_auc = auc(fpr, tpr)

        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_val, y_probs)
        pr_auc = auc(recall, precision)

        # Plot ROC curve
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc='lower right')

        # Plot Precision-Recall curve
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='green', lw=2, label='Precision-Recall curve (AUC = %0.2f)' % pr_auc)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} Precision-Recall Curve')
        plt.legend(loc='lower left')

        plt.tight_layout()
        plt.show()
    else:
        print(f"Model {model_name} does not support probability estimates.")
