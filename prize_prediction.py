import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


#Load Encoder And Decoder

with open('one_hot_encode_fuel_type.pkl','rb') as file:
    one_hot_fuel_type=pickle.load(file)

with open('one_hot_encode_body_type.pkl','rb') as file:
    one_hot_body_type=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


with open('label_encoder_make.pkl','rb') as file:
    label_encoder_make=pickle.load(file)

with open('label_encoder_model.pkl','rb') as file:
    label_encoder_model=pickle.load(file)

with open('label_encoder_transmission.pkl','rb') as file:
    label_encoder_transmission=pickle.load(file)

with open('label_encoder_condition.pkl','rb') as file:
    label_encoder_condition=pickle.load(file)

with open('decision_tree.pkl','rb') as file:
    decision_tree=pickle.load(file)



## Streamlit app

st.title('Car Prize Prediction')

##User Input

make=st.selectbox('Make',label_encoder_make.classes_)
model=st.selectbox('Model',label_encoder_model.classes_)
mileage=st.number_input('Mileage')
engine_hp=st.number_input('Engine_HP')
transmission=st.selectbox('Transmission',label_encoder_transmission.classes_)
fuel_type=st.selectbox('Fuel Type',one_hot_fuel_type.categories_[0])
body_type=st.selectbox('Body Type',one_hot_body_type.categories_[0])
condition=st.selectbox('Condition',label_encoder_condition.classes_)
vehicle_age=st.slider('Age of vehicle',0,30)
mileage_per_year=st.number_input('Mileage Per Year')
#price=st.number_input


#Prepare the input data
input_data=pd.DataFrame({
    'make':[label_encoder_make.transform([make])[0]],
    'model':[label_encoder_model.transform([model])[0]],
    'mileage':[mileage],
    'engine_hp':[engine_hp],
    'transmission':[label_encoder_transmission.transform([transmission])[0]],
    'condition':[label_encoder_condition.transform([condition])[0]],
    'vehicle_age':[vehicle_age],
    'mileage_per_year':[mileage_per_year]
})

fuel_type_encoded=one_hot_fuel_type.transform([[fuel_type]])
body_type_encoded=one_hot_body_type.transform([[body_type]])
fuel_type_df=pd.DataFrame(fuel_type_encoded,columns=one_hot_fuel_type.get_feature_names_out(['fuel_type']))
body_type_df=pd.DataFrame(body_type_encoded,columns=one_hot_body_type.get_feature_names_out(['body_type']))

encode_df=pd.concat([fuel_type_df,body_type_df],axis=1)


##combine final data
input_data=pd.concat([input_data.reset_index(drop=True),encode_df],axis=1)


##Scaler Transform

input_data_scaled=scaler.transform(input_data)



##Prediction
prediction=decision_tree.predict(input_data_scaled)

st.write(f"The car's Price is:{prediction}")