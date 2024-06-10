import streamlit as st
import pickle
import numpy as np
import pandas as pd
import gzip

st.title("Singapore Flat Resale Price Prediction")

col1, col2 = st.columns(2)

with col1:
   town = st.selectbox(
    "Town ",
    ('ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
       'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI',
       'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
       'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG',
       'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN',
       'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS',
       'PUNGGOL'))
   flat_type = st.selectbox(
    "Flat Type",
    ('1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE',
       'MULTI GENERATION', 'MULTI-GENERATION'))
   storey_range = st.selectbox(
    "Storey Range",
    ('10 TO 12', '04 TO 06', '07 TO 09', '01 TO 03', '13 TO 15',
       '19 TO 21', '16 TO 18', '25 TO 27', '22 TO 24', '28 TO 30',
       '31 TO 33', '40 TO 42', '37 TO 39', '34 TO 36', '06 TO 10',
       '01 TO 05', '11 TO 15', '16 TO 20', '21 TO 25', '26 TO 30',
       '36 TO 40', '31 TO 35', '46 TO 48', '43 TO 45', '49 TO 51'))
   flat_model = st.selectbox(
    "Flat Model",
    ('IMPROVED', 'NEW GENERATION', 'MODEL A', 'STANDARD', 'SIMPLIFIED',
       'MODEL A-MAISONETTE', 'APARTMENT', 'MAISONETTE', 'TERRACE',
       '2-ROOM', 'IMPROVED-MAISONETTE', 'MULTI GENERATION',
       'PREMIUM APARTMENT', 'Improved', 'New Generation', 'Model A',
       'Standard', 'Apartment', 'Simplified', 'Model A-Maisonette',
       'Maisonette', 'Multi Generation', 'Adjoined flat',
       'Premium Apartment', 'Terrace', 'Improved-Maisonette',
       'Premium Maisonette', '2-room', 'Model A2', 'DBSS', 'Type S1',
       'Type S2', 'Premium Apartment Loft', '3Gen'))

with col2:
  floor_area_sqm = st.number_input("Floor Area sqm",min_value=28.0, max_value=173.0)
  lease_commence_date = st.selectbox(
    "Lease Commence Date",
    (1977. , 1976. , 1978. , 1979. , 1984. , 1980. , 1985. , 1981. ,
       1982. , 1986. , 1972. , 1983. , 1973. , 1969. , 1975. , 1971. ,
       1974. , 1967. , 1970. , 1968. , 1988. , 1987. , 1989. , 1990. ,
       1992. , 1993. , 1994. , 1991. , 1995. , 1996. , 1997. , 1998. ,
       1999. , 2000. , 2001. , 1966. , 2002. , 2006. , 2003. , 2005. ,
       2004. , 2008. , 2007. , 2009. , 2010. , 2012. , 2011. , 2013. ,
       2014. , 2015. , 2016. , 2017. , 2018. , 2018.5))
  year = st.selectbox(
    "Year",
    (1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
       2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
       2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,
       2023, 2024))
  




new_data = pd.DataFrame({
  'town':[town],
  'flat_type':[flat_type],
  'storey_range':[storey_range],
  'flat_model':[flat_model],
  'floor_area_sqm':[floor_area_sqm],
  'lease_commence_date':[lease_commence_date],
  'year':[year]
})

# Load the encoder, scalers, and model from pickle files
category_col = ['town','flat_type','storey_range','flat_model']
numerical_col = ['floor_area_sqm','lease_commence_date','year']

with open('encoder.pkl', 'rb') as f:
  encoder = pickle.load(f)

with open('scaler_features.pkl', 'rb') as f:
  scaler_features = pickle.load(f)

with open('scaler_target.pkl', 'rb') as f:
  scaler_target = pickle.load(f)

with gzip.open("dtm.pkl.gz", "rb") as f:
  model = pickle.load(f)

# Preprocess new data (similar to the training process)
categorical_data = new_data[category_col]
numerical_data = new_data[numerical_col]

# Encode categorical data
encoded_categorical_data = encoder.transform(categorical_data)

# Scale numerical data
scaled_numerical_data = scaler_features.transform(numerical_data)

# Combine preprocessed data
preprocessed_data = np.concatenate((encoded_categorical_data, scaled_numerical_data), axis=1)

# Predict resale price
predicted_price = model.predict(preprocessed_data)  # Get the first prediction (assuming single data point)
predicted_price_scaled = np.array([predicted_price])
predicted_price_original_scale = scaler_target.inverse_transform(predicted_price_scaled)
resale_price = np.exp(predicted_price_original_scale) 

# st.button("PREDICT RESALE PRICE",type="primary")
if st.button("PREDICT RESALE PRICE",type="primary"):
    st.write("**Predicted resale price: {:.2f}**".format(resale_price[0, 0]))


