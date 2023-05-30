import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

def preprocess_data(data):
    try: 
        data = data.drop("customerID", axis=1)
    except:
        pass
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data = data.dropna() 
    return data 

def load_pickles(model_pickle_path): #, transformer_pickle_path):
    with open(model_pickle_path, "rb") as file:
        model = pickle.load(file)
    #with open(transformer_pickle_path, "rb") as file:
    #    transform_data = pickle.load(file)
    return model  #, transform_data


def transform_data(train, val):

    train_num_cols = train.loc[:, ['tenure', 'MonthlyCharges', 'TotalCharges']]
    train_cat_cols = train.drop(['tenure', 'MonthlyCharges', 'TotalCharges','Churn'], axis=1)
    y_train = train['Churn'].map({'Yes':1,'No':0})

    # OHE categorical
    ohe = OneHotEncoder(sparse=False).fit(train_cat_cols)
    encoded_train = ohe.transform(train_cat_cols)
    encoded_df_train = pd.DataFrame(encoded_train, columns=ohe.get_feature_names_out())
    # Scale numerical 
    scaler = MinMaxScaler().fit(train_num_cols) 
    scaled_train = scaler.transform(train_num_cols)
    scaled_df_train = pd.DataFrame(scaled_train, columns=scaler.get_feature_names_out())

    X_train = pd.concat([encoded_df_train, scaled_df_train], axis=1)

    val_num_cols = val.loc[:, ['tenure', 'MonthlyCharges', 'TotalCharges']]
    val_cat_cols = val.drop(['tenure', 'MonthlyCharges', 'TotalCharges','Churn'], axis=1)
    y_val = val['Churn'].map({'Yes':1,'No':0})

    encoded_val = ohe.transform(val_cat_cols)
    encoded_df_val = pd.DataFrame(encoded_val, columns=ohe.get_feature_names_out())

    scaled_val = scaler.transform(val_num_cols)
    scaled_df_val = pd.DataFrame(scaled_val, columns=scaler.get_feature_names_out())

    X_val = pd.concat([encoded_df_val, scaled_df_val], axis=1)
        
    return X_train, y_train, X_val, y_val 


def make_predictions(train, test):
    model_pickle_path = "./models/churn_pred_model.pkl"
    transformer_pickle_path = "./models/churn_pred_label_encoder.pkl"
    model = load_pickles(model_pickle_path) #, transformer_pickle_path)

    train = preprocess_data(train)
    test = preprocess_data(test)
    
    X_train, X_test, y_train, y_test = transform_data(train, test)
    prediction = model.predict(y_train)
    
    return prediction 




if __name__ == '__main__':
    #st.title("Customer Churn Prediction")

    # get  data
    train = pd.read_csv("./data/training_data.csv")

    df = pd.read_csv("./data/training_data.csv")
    df = preprocess_data(df)

    st.title('New Customer Churn Prediction')
    # Create input fields for categorical variables
    gender = st.selectbox('Gender', df['gender'].unique())
    senior_citizen = st.selectbox('Senior Citizen', ['Yes', 'No']) #df['SeniorCitizen'].unique())
    if senior_citizen == 'Yes':
        senior_citizen = 1
    else: 
        senior_citizen = 0
    partner = st.selectbox('Partner', df['Partner'].unique())
    dependents = st.selectbox('Dependents', df['Dependents'].unique())
    # Create input field for numerical variable
    tenure = st.slider('Tenure', min_value=0, max_value=100)
    # Create input fields for other variables
    phone_service = st.selectbox('Phone Service', df['PhoneService'].unique())
    multiple_lines = st.selectbox('Multiple Lines', df['MultipleLines'].unique())
    internet_service = st.selectbox('Internet Service', df['InternetService'].unique())
    online_security = st.selectbox('Online Security', df['OnlineSecurity'].unique())
    online_backup = st.selectbox('Online Backup', df['OnlineBackup'].unique())
    device_protection = st.selectbox('Device Protection', df['DeviceProtection'].unique())
    tech_support = st.selectbox('Tech Support', df['TechSupport'].unique())
    streaming_tv = st.selectbox('Streaming TV', df['StreamingTV'].unique())
    streaming_movies = st.selectbox('Streaming Movies', df['StreamingMovies'].unique())
    contract = st.selectbox('Contract', df['Contract'].unique())
    paperless_billing = st.selectbox('Paperless Billing', df['PaperlessBilling'].unique())
    payment_method = st.selectbox('Payment Method', df['PaymentMethod'].unique())

    # Create input field for numerical variables
    monthly_charges = st.slider('Monthly Charges', min_value=0, max_value=200)
    total_charges = st.slider('Total Charges', min_value=0, max_value=10000)

    # Create button to add the new sample to the DataFrame
    new_sample = {
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
            }

    test = pd.DataFrame([new_sample])
    st.dataframe(test)
    test['Churn'] = 'No'
            
    if st.button("Predict Churn"):
        pred = make_predictions(train, test)
        #st.text(f"Prediction is: {pred[0]}")  
        if pred[0] == 1:
            st.text("The Customer is predicted to CHURN")
        else:
            st.text("The Customer is predicted to NOT CHURN")
