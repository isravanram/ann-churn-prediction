import streamlit as st
from churn_prediction_app import import_parameters,predict_churn_probability


label_encoder,one_hot_encoder,standard_scaler,model = import_parameters()
parameters = [label_encoder,one_hot_encoder,standard_scaler,model]
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography',one_hot_encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder.classes_)
age = st.slider('Age',18,80)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

input_data ={
      'CreditScore':credit_score,
      'Geography':geography,
      'Gender':gender,
      'Age':age,
      'Tenure':tenure,
      'Balance':balance,
      'NumOfProducts':credit_score,
      'HasCrCard':has_cr_card,
      'IsActiveMember':is_active_member,
      'EstimatedSalary':estimated_salary
}
prediction,predict_proba = predict_churn_probability(input_data,parameters)

st.write(prediction)