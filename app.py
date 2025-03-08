import streamlit as st
# Importing important libraries
import pandas as pd
from pyexpat import features
from  sklearn.preprocessing import LabelEncoder
import numpy as np
from lightgbm import LGBMRegressor
import pickle
#import matplotlib.pyplot as plt
#import seaborn as sns
from datetime import datetime
from datetime import date as dt
#import plotly.express as px
#import plotly.graph_objects as go
import warnings

from streamlit import selectbox

warnings.filterwarnings('ignore')
model=pickle.load(open('model_lgbm.pkl','rb'))

st.set_page_config(
    page_title="Delivery Time Predictor",
    page_icon="🚴‍♀️",
    layout="wide"
)
st.title("Zomato delivery duration Estimator")
col1,col2,col3=st.columns(3)
with col1:
    age = st.number_input("delivery_person_age", min_value=18, max_value=50, value=25)

with col2:
    delivery_person_ratings =st.number_input("delivery_person_ratings", min_value=0.0, max_value=5.0, value=3.0)

with col3:
    weather_condition=st.selectbox('weather_condition',['Fog','Stormy','Cloudy','Sandstorms','Windy','Sunny'])
with col1:
    road_traffic_density=st.selectbox('road_traffic_density',['Low','Medium','High','Jam'])

with col2:
    vehicle_condition=st.selectbox('vehicle_condition',['Poor Condition','Moderate Condition','Good Condition','Rare Condition'])

with col3:
    type_of_vehicle=st.selectbox('type_of_vehicle',['motorcycle', "scooter", "electric_scooter"])

with col1:
    multiple_deliveries=st.selectbox('multiple_deliveries',[0.0,1.0,2.0,3.0])

with col3:
    festival=st.selectbox('festival',['No','Yes'])

with col1:
    city=st.selectbox('city',['Metropolitian','Urban','Semi-Urban'])

with col2:
    distance_category=st.selectbox('distance_category',['Very Short(less than 2km)','Short(Between 2 to 5km)','Medium(5 to 10km)','Long(Between 10 to 20km)','Very Long'])

vehicle_condition_map = {'Poor Condition': 0, 'Moderate Condition': 1, 'Good Condition': 2, 'Rare Condition': 3}
vehicle_condition = vehicle_condition_map.get(vehicle_condition, vehicle_condition)

distance_category_map = {
    'Very Short(less than 2km)': 0,
    'Short(Between 2 to 5km)': 1,
    'Medium(5 to 10km)': 2,
    'Long(Between 10 to 20km)': 3,
    'Very Long': 4
}
distance_category = distance_category_map.get(distance_category, distance_category)

# one_hot_col = ['weather_conditions', 'festival', 'city']
# le=LabelEncoder()
# for col in one_hot_col:
#      #locals()[col]=le.fit_transform([locals()[col]])[0]
#      locals()[col] = le.fit_transform([locals()[col]])[0]
one_hot_col = ['weather_condition', 'festival', 'city']
le = LabelEncoder()
for col in one_hot_col:
    locals()[col] = le.fit_transform([locals()[col]])[0]


type_of_vehicle_map={'motorcycle':0, "scooter":1, "electric_scooter":2}
type_of_vehicle=type_of_vehicle_map.get(type_of_vehicle,type_of_vehicle)

def label_encod(x):
    if x == 'Jam':
        return 4
    elif x == 'High':
        return 3
    elif x == 'Medium':
        return 2
    else:
        return 1

road_traffic_density =label_encod(road_traffic_density)

if st.button("Check the Taken Time"):

    features = np.array([[age, delivery_person_ratings, weather_condition, road_traffic_density, vehicle_condition,
                          type_of_vehicle, multiple_deliveries, festival, city, distance_category]])

    prediction= model.predict(features)

    st.write(f"For this delivery, the predicted time taken is: {prediction[0]} minutes" )