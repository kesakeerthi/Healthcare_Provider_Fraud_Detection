import streamlit as st
import pandas as pd
import pickle
from src.preprocess import preprocess

# Loading XG Boost which is trained and tuned with Dataset
model = pickle.load(open('D:/ML_Projects/Healthcare_Provider_Fraud_Detection/pickles/xg_boost.pkl','rb'))

# Loading Standard Scaler model to scale the data
scaler = pickle.load(open('D:/ML_Projects/Healthcare_Provider_Fraud_Detection/pickles/standard_scaler.pkl','rb'))

# Loading Encoding Dicts used while training for State, County, Race
County_Encoded = pickle.load(open('D:/ML_Projects/Healthcare_Provider_Fraud_Detection/pickles/County_Encoded.pkl','rb'))
State_Encoded = pickle.load(open('D:/ML_Projects/Healthcare_Provider_Fraud_Detection/pickles/State_Encoded.pkl','rb'))
Race_Encoded = pickle.load(open('D:/ML_Projects/Healthcare_Provider_Fraud_Detection/pickles/Race_Encoded.pkl','rb'))


def predict_fraud(X):
    '''This function takes details about a healthcare provider as input and returns a prediction of the healthcare provider
       being a potential fraud.'''

    # Scaling data
    X_scaled = scaler.transform(X)


    # Predictions
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)

    y_pred_prob =  y_prob[:, 1]

    # Initialize data to Dicts of series.
    df = {'y_pred': y_pred, 'y_prob':y_pred_prob  }

    return df

def main():
    html_temp = """
    <div style="background-color:DodgerBlue;padding:10px">
    <h2 style="color:white;text-align:center;">Healthcare Provider Fraud Predictor </h2>
    </div>
    <br>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.write("Please upload file of the claim to know if related providers have done fraud or not.")

    st.subheader("Dataset")
    data_file = st.file_uploader("Upload CSV", type=["csv"])


    if data_file is not None:
        file_details = {"filename": data_file.name, "filetype": data_file.type,  "filesize": data_file.size}

        st.write(file_details)
        df = pd.read_csv(data_file, index_col=0)
        df_input = df.copy()
        st.dataframe(df)

        df['Race_Encoded'] = df['Race'].apply(lambda x: Race_Encoded.get(x))
        df['State_Encoded'] = df['State'].apply(lambda x: State_Encoded.get(x))
        df['County_Encoded'] = df['County'].apply(lambda x: County_Encoded.get(x))

        X = preprocess(df)

        if st.button("Predict"):
            predictions = predict_fraud(X)
            predictions = pd.DataFrame.from_dict(predictions)
            st.dataframe(predictions)

            df.drop(['Race_Encoded', 'State_Encoded', 'County_Encoded'], axis = 1, inplace=True)
            Result = pd.concat([df_input, predictions], axis=1)

            st.write('To Download Input File with Corresponding Predictions: Download CSV')
            st.download_button(label='Download CSV', data=Result.to_csv(index=False), mime='text/csv', file_name='Results.csv')