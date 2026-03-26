# =========================
# IMPORTS
# =========================
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/flightprices.csv")


# =========================
# FEATURE ENGINEERING
# =========================

# Date
df["Date"] = df["Date_of_Journey"].str.split('/').str[0].astype(int)
df["Month"] = df["Date_of_Journey"].str.split('/').str[1].astype(int)
df["Year"] = df["Date_of_Journey"].str.split('/').str[2].astype(int)
df.drop("Date_of_Journey", axis=1, inplace=True)

# Stops
stop_mappings = {
    'non-stop': 0,
    '1 stop': 1,
    '2 stops': 2,
    '3 stops': 3,
    '4 stops': 4
}
df["Total_Stops"] = df["Total_Stops"].map(stop_mappings).fillna(1).astype(int)

# Departure time
df["Dep_hour"] = df["Dep_Time"].str.split(":").str[0].astype(int)
df["Dep_min"] = df["Dep_Time"].str.split(":").str[1].astype(int)
df.drop("Dep_Time", axis=1, inplace=True)

# Arrival time
df["Arrival_hour"] = df["Arrival_Time"].str.split(':').str[0].astype(int)
df["Arrival_min"] = df["Arrival_Time"].str.split(':').str[1].str.split(' ').str[0].astype(int)

month_mappings = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

df["Arrival_date"] = pd.to_numeric(
    df["Arrival_Time"].str.split(' ').str.get(1),
    errors='coerce'
).fillna(0).astype(int)

df["Arrival_month"] = df["Arrival_Time"].str.split(' ').str.get(2).map(month_mappings).fillna(0).astype(int)

df.drop("Arrival_Time", axis=1, inplace=True)

# Source
source_mappings = {
    'Banglore': 0, 'Cochin': 1, 'Delhi': 2,
    'Kolkata': 3, 'Mumbai': 4, 'Chennai': 5
}
df["Source"] = df["Source"].map(source_mappings).astype(int)

# Destination
destination_mappings = {
    'Banglore': 0, 'Cochin': 1, 'New Delhi': 2,
    'Delhi': 2, 'Kolkata': 3, 'Hyderabad': 4
}
df["Destination"] = df["Destination"].map(destination_mappings).fillna(-1).astype(int)

# Airline
airline_mappings = {
    'IndiGo': 1, 'Jet Airways': 3, 'Air India': 2,
    'SpiceJet': 4, 'Multiple carriers': 5, 'GoAir': 6,
    'Vistara': 7, 'Air Asia': 8,
    'Vistara Premium economy': 9, 'Jet Airways Business': 10,
    'Multiple carriers Premium economy': 11, 'Trujet': 12
}
df['Airline'] = df['Airline'].map(airline_mappings).fillna(-1).astype(int)

# Route → Stops
df["Computed_Stops"] = df["Route"].fillna("").apply(lambda x: len(str(x).split('→')) - 1)
df.drop("Route", axis=1, inplace=True)

# Duration
df['Duration'] = df['Duration'].str.replace('h', '*60').str.replace(' ', '+').str.replace('m', '*1').apply(eval)

# Additional Info
addinfo_mappings = {
    'No info': 0, 'No Info': 0,
    'In-flight meal not included': 1,
    'No check-in baggage included': 2,
    '1 Short layover': 3,
    '1 Long layover': 4,
    'Business class': 5,
    'Change airports': 6,
    'Red-eye flight': 7,
    '2 Long layover': 8,
}
df['Additional_Info'] = df['Additional_Info'].map(addinfo_mappings).fillna(-1).astype(int)


# =========================
# FINAL CHECK
# =========================
print("Remaining object columns:", df.select_dtypes(include='object').columns)


# =========================
# MODEL TRAINING
# =========================

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline models
lr = LinearRegression()
lr.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluation
lr_preds = lr.predict(X_test)
rf_preds = rf.predict(X_test)

print("\nLinear Regression:")
print("MAE:", mean_absolute_error(y_test, lr_preds))
print("R2:", r2_score(y_test, lr_preds))

print("\nRandom Forest:")
print("MAE:", mean_absolute_error(y_test, rf_preds))
print("R2:", r2_score(y_test, rf_preds))


# =========================
# HYPERPARAMETER TUNING (FAST)
# =========================

param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=10,      # reduce to 5 if slow
    cv=3,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)

print("\n⏳ Running RandomizedSearchCV...")

random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_

best_preds = best_rf.predict(X_test)

print("\n🔥 Tuned Random Forest:")
print("MAE:", mean_absolute_error(y_test, best_preds))
print("R2:", r2_score(y_test, best_preds))
print("Best Params:", random_search.best_params_)


# =========================
# SAVE MODEL
# =========================

pickle.dump(best_rf, open("model/model.pkl", "wb"))
pickle.dump(X.columns, open("model/columns.pkl", "wb"))

print("\n✅ Model saved successfully!")