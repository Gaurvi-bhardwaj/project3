import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('mobile_price_model.pkl')

# Streamlit app
st.title('Mobile Price Prediction')

# Input fields
battery = st.number_input('Total energy a battery can store in one time measured in mAh', min_value=0, step=10)
bluetooth = st.selectbox('Has bluetooth or not', ['No', 'Yes'])
processor_speed = st.number_input('Speed at which microprocessor executes instructions', min_value=0.0, step=0.1)
dual_sim = st.selectbox('Has dual sim support or not', ['No', 'Yes'])
front_camera = st.number_input('Front Camera mega pixels', min_value=0.0, step=0.1)
support_4g = st.selectbox('Has 4G or not', ['No', 'Yes'])
internal_memory = st.number_input('Internal Memory in Gigabytes', min_value=0, step=1)
depth = st.number_input('Mobile Depth in cm', min_value=0.0, step=0.1)
weight = st.number_input('Weight of mobile phone', min_value=0.0, step=0.1)
cores = st.number_input('Number of cores of processor', min_value=1, step=1)

# Convert categorical inputs to numerical values
bluetooth = 1 if bluetooth == 'Yes' else 0
dual_sim = 1 if dual_sim == 'Yes' else 0
support_4g = 1 if support_4g == 'Yes' else 0

# Predict button
if st.button('Predict Price'):
    # Prepare the input array
    input_data = np.array([[battery, bluetooth, processor_speed, dual_sim, front_camera, support_4g, internal_memory, depth, weight, cores]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the result
    st.write(f'Estimated Mobile Price: ${prediction[0]:.2f}')

# Run the app with: streamlit run app.py
