import json
import altair as alt
import pandas as pd
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import col
import streamlit as st
import sys
import cachetools
import joblib
from snowflake.snowpark import types as T
from snowflake.snowpark import functions as F

st.set_page_config(
    page_title="Assignment 3 Part 2",
    page_icon=":smiley:",
    layout="wide"
)
APP_ICON_URL = "https://as1.ftcdn.net/v2/jpg/01/85/75/82/1000_F_185758274_tyBRRmslE10iCmSF9bRPsaZXgF7QUiLE.jpg"

# Session create


def create_session():
    if "snowpark_session" not in st.session_state:
        session = Session.builder.configs(
            json.load(open("connection.json"))).create()
        st.session_state['snowpark_session'] = session
    else:
        session = st.session_state['snowpark_session']
    return session


# Snowpark initialization
st.write(
    "<style>[data-testid='stMetricLabel'] {min-height: 0.5rem !important}</style>", unsafe_allow_html=True)
st.image(APP_ICON_URL, width=80)
st.title("Predict Customer Lifetime Value ")
session = create_session()
session.sql_simplifier_enabled = True
session.add_import("@ml_models/model.joblib")
session.add_packages('snowflake-snowpark-python', 'scikit-learn',
                     'pandas', 'numpy', 'joblib', 'cachetools', 'xgboost', 'joblib')


# Define the list of features and their data types
st.header("Input Features")
features = ['C_BIRTH_YEAR', 'CA_ZIP', 'CD_GENDER', 'CD_MARITAL_STATUS',
            'CD_CREDIT_RATING', 'CD_EDUCATION_STATUS', 'CD_DEP_COUNT']
data_types = [int, str, str, str, str, str, int]

# Define inputs
inputs = []
birth_year_range = (1960, 2020)
gender_options = ['Male', 'Female']
marital_status_options = ['Married', 'Widowed', 'Single', 'Divorced']
credit_rating_options = ["Good", "High Risk", "Unknown"]
education_status_options = ['Advanced Degree', 'Unknown',
                            '4 yr Degree', 'Primary', '2 yr Degree', 'Secondary', 'College']

# User prompts for input
inp, out = st.columns(2)
for i, feature in enumerate(features):
    if data_types[i] == int:
        if feature == 'C_BIRTH_YEAR':
            inputs.append(int(inp.slider(
                f'Enter {feature}', min_value=birth_year_range[0], max_value=birth_year_range[1])))
        else:
            dep_count = inp.selectbox(f'Enter {feature}', options=['0', '1'])
            if (dep_count == '1'):
                inputs.append(1)
            else:
                inputs.append(0)
    else:
        if feature == 'CA_ZIP':
            inputs.append(inp.text_input(f'Enter {feature}', max_chars=5, value="02130"))
        elif feature == 'CD_GENDER':
            gender_input = inp.selectbox(
                f'Enter {feature}', options=gender_options)
            if gender_input == 'Male':
                inputs.append('M')
            else:
                inputs.append('F')
        elif feature == 'CD_MARITAL_STATUS':
            marital_status = inp.selectbox(
                f'Enter {feature}', options=marital_status_options)
            if gender_input == 'Married':
                inputs.append('M')
            elif gender_input == 'Widowed':
                inputs.append('W')
            elif gender_input == 'Single':
                inputs.append('S')
            else:
                inputs.append('D')

        elif feature == 'CD_CREDIT_RATING':
            inputs.append(inp.selectbox(
                f'Enter {feature}', options=credit_rating_options))
        else:
            inputs.append(inp.selectbox(
                f'Enter {feature}', options=education_status_options))

# Show user inputs as DF
df = session.create_dataframe([inputs], schema=features)
out.dataframe(df)
st.columns([2,1])
# Call predict function on button click
if st.button('Predict CLV'):
    out.write('Intializing SnowPark...')
    df = session.create_dataframe([inputs], schema=features)
    
    # Add util funcs to call UDF on user input
    @cachetools.cached(cache={})
    def read_file(filename):
        import os
        import joblib
        import_dir = sys._xoptions.get("snowflake_import_directory")
        if import_dir:
            with open(os.path.join(import_dir, filename), 'rb') as file:
                m = joblib.load(file)
                return m

    @F.pandas_udf(session=session, max_batch_size=10000, is_permanent=True, stage_location='@ml_models', replace=True, name="clv_xgboost_udf")
    def predict(df:  T.PandasDataFrame[int, str, str, str, str, str, int]) -> T.PandasSeries[float]:
        m = read_file('model.joblib')
        df.columns = features
        return m.predict(df)

    @st.cache_data(show_spinner=False)
    def predictCLV(_df):
        df_predicted_clv = df.select(*df,
                                     predict(*df).alias('PREDICTION'))
        print("HERE", df_predicted_clv)
        df_predicted_clv = pd.DataFrame(df_predicted_clv.collect()).T
        return df_predicted_clv

    out.write('Intialized')

    out.write('Calling Snowpark XGBoost UDF on user given input ')
    df_predicted_clv = predictCLV(df)
    out.header('Predicted CLV for User Given Inputs')
    out.dataframe(df_predicted_clv)
    out.header(f"This customer is expected to invest a total of ${int(df_predicted_clv.loc['PREDICTION',0])} in our company")