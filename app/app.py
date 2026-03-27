import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("model/model.pkl", "rb"))
cols = pickle.load(open("model/columns.pkl", "rb"))

st.title("✈️ Flight Price Predictor")

st.write("Enter flight details:")

airline = st.selectbox("Airline", [
    "IndiGo", "Air India", "Jet Airways", "SpiceJet"
])

source = st.selectbox("Source", [
    "Delhi", "Kolkata", "Mumbai", "Chennai", "Banglore"
])

destination = st.selectbox("Destination", [
    "Cochin", "Delhi", "New Delhi", "Hyderabad", "Kolkata"
])

stops = st.selectbox("Total Stops", [0, 1, 2, 3])

duration = st.number_input("Duration (minutes)", min_value=30, max_value=2000)

dep_hour = st.slider("Departure Hour", 0, 23)
dep_min = st.slider("Departure Minute", 0, 59)

arrival_hour = st.slider("Arrival Hour", 0, 23)
arrival_min = st.slider("Arrival Minute", 0, 59)

airline_map = {
    "IndiGo": 1,
    "Air India": 2,
    "Jet Airways": 3,
    "SpiceJet": 4
}

source_map = {
    "Banglore": 0,
    "Cochin": 1,
    "Delhi": 2,
    "Kolkata": 3,
    "Mumbai": 4,
    "Chennai": 5
}

destination_map = {
    "Banglore": 0,
    "Cochin": 1,
    "New Delhi": 2,
    "Delhi": 2,
    "Kolkata": 3,
    "Hyderabad": 4
}

if st.button("Predict Price"):

    input_data = pd.DataFrame([0]*len(cols)).T
    input_data.columns = cols

    input_data["Airline"] = airline_map.get(airline, -1)
    input_data["Source"] = source_map.get(source, -1)
    input_data["Destination"] = destination_map.get(destination, -1)

    input_data["Total_Stops"] = stops
    input_data["Duration"] = duration

    input_data["Dep_hour"] = dep_hour
    input_data["Dep_min"] = dep_min

    input_data["Arrival_hour"] = arrival_hour
    input_data["Arrival_min"] = arrival_min

    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Price: ₹{int(prediction[0])}")