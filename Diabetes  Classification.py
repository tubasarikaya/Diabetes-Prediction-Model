###########
# Libraries & Column Settings
###########
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


###########
# Read Data
###########
df = pd.read_csv("C:\Users\Tuvba\Desktop\Machine Learning Eğitimi Turkishe\diabetes.csv")

df.head()

###########
# Overview
###########
def check_df(dataframe, head=5):
    print(" SHAPE ".center(70, "#"))
    print(dataframe.shape)
    print(" INFO ".center(70, "#"))
    print(dataframe.info())
    print(" MEMORY USAGE ".center(70, "#"))
    print(f"{dataframe.memory_usage().sum() / (1024 ** 2):.2f} MB")
    print(" NUNIQUE ".center(70, "#"))
    print(dataframe.nunique())
    print(" MISSING VALUES ".center(70, "#"))
    print(dataframe.isnull().sum())
    print(" DUPLICATED VALUES ".center(70, "#"))
    print(dataframe.duplicated().sum())

check_df(df)


percentiles = [0.10, 0.25, 0.35, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
df.describe(percentiles = percentiles).T


columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
zero_counts = {col: (df[col] == 0).sum() for col in columns_to_check}
for col, count in zero_counts.items():
    print(f"Number of observations with value 0 in '{col}': {count}")



## replace with nan
variables_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[variables_to_replace] = df[variables_to_replace].replace(0, np.nan)

check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "object" or dataframe[col].dtypes.name == "category"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "object" and dataframe[col].dtypes.name != "category"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   (dataframe[col].dtypes == "object" or dataframe[col].dtypes.name == "category")]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "object" and dataframe[col].dtypes.name != "category"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)


df["Outcome"].value_counts()

def num_summary(dataframe, numerical_col, plot=False):
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
for col in num_cols:
     num_summary(df, col, plot=False)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)
correlation_matrix(df, num_cols)



###########
# Preprocessing
###########

df.isnull().sum()

# missing values
columns_to_fill = ['Insulin', 'SkinThickness', 'BloodPressure', 'BMI', 'Glucose']
for column in columns_to_fill:
    df[column].fillna(df[column].median(), inplace=True)


# outliers
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Check for outliers before-after replacing
outlier_results_after = {col: check_outlier(df, col) for col in df.columns}
for col, has_outliers in outlier_results_after.items():
    print(f"{col}: {'Outliers present' if has_outliers else 'No outliers'}")


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in df.columns:
    replace_with_thresholds(df, col)


check_df(df)

cols = [col for col in df.columns if df[col].nunique() > 2]
scaler = StandardScaler()
df[cols] = scaler.fit_transform(df[cols])
df.head()

###########
# Base Model
###########

X = df.drop('Outcome', axis=1)
y = df['Outcome']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)


def evaluate_model(model, X_test, y_test, y_pred):
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
evaluate_model(log_model, X_test, y_test, y_pred)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


def plot_feature_importance(model, feature_names):
    importance = abs(model.coef_[0])

    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title("Logistic Regression - Feature Importance")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()
feature_names = df.drop('Outcome', axis=1).columns.tolist()
plot_feature_importance(log_model, feature_names)


def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt = ".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y_test, y_pred)


