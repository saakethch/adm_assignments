import json
import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
import sys
import cachetools

from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import col
from snowflake.snowpark import types as T
from snowflake.snowpark import functions as F
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
import sqlalchemy
import plotly.express as px
import os
import joblib
st.set_page_config(page_title="Assignment 3", layout="wide")
APP_ICON_URL = "https://as1.ftcdn.net/v2/jpg/01/85/75/82/1000_F_185758274_tyBRRmslE10iCmSF9bRPsaZXgF7QUiLE.jpg"
st.write(
    "<style>[data-testid='stMetricLabel'] {min-height: 0.5rem !important}</style>", unsafe_allow_html=True)
st.image(APP_ICON_URL, width=80)
st.title("TPCDS Data Exploration and Analysis")

# Session create
def create_session():
    if "snowpark_session" not in st.session_state:
        session = Session.builder.configs(json.load(open("connection.json"))).create()
        st.session_state["snowpark_session"] = session
    else:
        session = st.session_state["snowpark_session"]
    return session


def part1():
    tables = ["query2", "query3", "query4", "query5"]
    engine = create_engine(
        URL(
            account="rn99162.us-east4.gcp",
            user="saaketh",
            password="Saaketh12#",
            database="SNOWFLAKE_SAMPLE_DATA",
            schema="TPCDS_SF10TCL",
            warehouse="COMPUTE_WH",
            role="ACCOUNTADMIN",
        )
    )

    questions = [
        "Compute the average quantity, list price, discount, and sales price for promotional items sold in stores where the promotion is not offered by mail or a special event. Restrict the results to a specific gender, marital and educational status.",
        "Compute the net profit of stores located in 400 Metropolitan areas with more than 10 preferred customers.",
        "Categorize store sales transactions into 5 buckets according to the number of items sold. Each bucket contains",
        "The average discount amount, sales price, list price, tax, net paid, paid price including tax, or net profit.",
        "Count the customers with the same gender, marital status, education status, purchase estimate, credit rating, dependent count, employed dependent count and college dependent count who live in certain counties and who have purchased from both stores and another sales channel during a three month time period of a given year. ",
    ]

    l, r = st.columns(2)
    st.columns([2, 1])

    # query 1
    r.write("Query 1 inputs")
    q1month = r.selectbox(
        "Select a month",
        range(1, 13),
    )
    q1year = r.selectbox(
        "Select a year",
        range(1980, 2020),
    )

    # query 2

    # query 3

    queries = [
        """ select  i_item_id, 
        avg(ss_quantity) agg1,
        avg(ss_list_price) agg2,
        avg(ss_coupon_amt) agg3,
        avg(ss_sales_price) agg4 
 from store_sales, customer_demographics, date_dim, item, promotion
 where ss_sold_date_sk = d_date_sk and
       ss_item_sk = i_item_sk and
       ss_cdemo_sk = cd_demo_sk and
       ss_promo_sk = p_promo_sk and
       cd_gender = 'M' and 
       cd_marital_status = 'S' and
       cd_education_status = 'College' and
       (p_channel_email = 'N' or p_channel_event = 'N') and
       d_year = 2001 
 group by i_item_id
 order by i_item_id
  limit 100;
""",
        """
 select  s_store_name
      ,sum(ss_net_profit)
 from store_sales
     ,date_dim
     ,store,
     (select ca_zip
     from (
      SELECT substr(ca_zip,1,5) ca_zip
      FROM customer_address
      WHERE substr(ca_zip,1,5) IN (
                          '24128','76232','65084','87816','83926','77556',
                          '20548','26231','43848','15126','91137',
                          '61265','98294','25782','17920','18426',
                          '98235','40081','84093','28577','55565',
                          '17183','54601','67897','22752','86284',
                          '18376','38607','45200','21756','29741',
                          '96765','23932','89360','29839','25989',
                          '28898','91068','72550','10390','18845',
                          '47770','82636','41367','76638','86198',
                          '81312','37126','39192','88424','72175',
                          '81426','53672','10445','42666','66864',
                          '66708','41248','48583','82276','18842',
                          '78890','49448','14089','38122','34425',
                          '79077','19849','43285','39861','66162',
                          '77610','13695','99543','83444','83041',
                          '12305','57665','68341','25003','57834',
                          '62878','49130','81096','18840','27700',
                          '23470','50412','21195','16021','76107',
                          '71954','68309','18119','98359','64544',
                          '10336','86379','27068','39736','98569',
                          '28915','24206','56529','57647','54917',
                          '42961','91110','63981','14922','36420',
                          '23006','67467','32754','30903','20260',
                          '31671','51798','72325','85816','68621',
                          '13955','36446','41766','68806','16725',
                          '15146','22744','35850','88086','51649',
                          '18270','52867','39972','96976','63792',
                          '11376','94898','13595','10516','90225',
                          '58943','39371','94945','28587','96576',
                          '57855','28488','26105','83933','25858',
                          '34322','44438','73171','30122','34102',
                          '22685','71256','78451','54364','13354',
                          '45375','40558','56458','28286','45266',
                          '47305','69399','83921','26233','11101',
                          '15371','69913','35942','15882','25631',
                          '24610','44165','99076','33786','70738',
                          '26653','14328','72305','62496','22152',
                          '10144','64147','48425','14663','21076',
                          '18799','30450','63089','81019','68893',
                          '24996','51200','51211','45692','92712',
                          '70466','79994','22437','25280','38935',
                          '71791','73134','56571','14060','19505',
                          '72425','56575','74351','68786','51650',
                          '20004','18383','76614','11634','18906',
                          '15765','41368','73241','76698','78567',
                          '97189','28545','76231','75691','22246',
                          '51061','90578','56691','68014','51103',
                          '94167','57047','14867','73520','15734',
                          '63435','25733','35474','24676','94627',
                          '53535','17879','15559','53268','59166',
                          '11928','59402','33282','45721','43933',
                          '68101','33515','36634','71286','19736',
                          '58058','55253','67473','41918','19515',
                          '36495','19430','22351','77191','91393',
                          '49156','50298','87501','18652','53179',
                          '18767','63193','23968','65164','68880',
                          '21286','72823','58470','67301','13394',
                          '31016','70372','67030','40604','24317',
                          '45748','39127','26065','77721','31029',
                          '31880','60576','24671','45549','13376',
                          '50016','33123','19769','22927','97789',
                          '46081','72151','15723','46136','51949',
                          '68100','96888','64528','14171','79777',
                          '28709','11489','25103','32213','78668',
                          '22245','15798','27156','37930','62971',
                          '21337','51622','67853','10567','38415',
                          '15455','58263','42029','60279','37125',
                          '56240','88190','50308','26859','64457',
                          '89091','82136','62377','36233','63837',
                          '58078','17043','30010','60099','28810',
                          '98025','29178','87343','73273','30469',
                          '64034','39516','86057','21309','90257',
                          '67875','40162','11356','73650','61810',
                          '72013','30431','22461','19512','13375',
                          '55307','30625','83849','68908','26689',
                          '96451','38193','46820','88885','84935',
                          '69035','83144','47537','56616','94983',
                          '48033','69952','25486','61547','27385',
                          '61860','58048','56910','16807','17871',
                          '35258','31387','35458','35576')
     intersect
      select ca_zip
      from (SELECT substr(ca_zip,1,5) ca_zip,count(*) cnt
            FROM customer_address, customer
            WHERE ca_address_sk = c_current_addr_sk and
                  c_preferred_cust_flag='Y'
            group by ca_zip
            having count(*) > 10)A1)A2) V1
 where ss_store_sk = s_store_sk
  and ss_sold_date_sk = d_date_sk
  and d_qoy = 2 and d_year = 1998
  and (substr(s_zip,1,2) = substr(V1.ca_zip,1,2))
 group by s_store_name
 order by s_store_name
  limit 100;
""",
        """select case when (select count(*) 
                  from store_sales 
                  where ss_quantity between 1 and 20) > 74129
            then (select avg(ss_ext_discount_amt) 
                  from store_sales 
                  where ss_quantity between 1 and 20) 
            else (select avg(ss_net_paid)
                  from store_sales
                  where ss_quantity between 1 and 20) end bucket1 ,
       case when (select count(*)
                  from store_sales
                  where ss_quantity between 21 and 40) > 122840
            then (select avg(ss_ext_discount_amt)
                  from store_sales
                  where ss_quantity between 21 and 40) 
            else (select avg(ss_net_paid)
                  from store_sales
                  where ss_quantity between 21 and 40) end bucket2,
       case when (select count(*)
                  from store_sales
                  where ss_quantity between 41 and 60) > 56580
            then (select avg(ss_ext_discount_amt)
                  from store_sales
                  where ss_quantity between 41 and 60)
            else (select avg(ss_net_paid)
                  from store_sales
                  where ss_quantity between 41 and 60) end bucket3,
       case when (select count(*)
                  from store_sales
                  where ss_quantity between 61 and 80) > 10097
            then (select avg(ss_ext_discount_amt)
                  from store_sales
                  where ss_quantity between 61 and 80)
            else (select avg(ss_net_paid)
                  from store_sales
                  where ss_quantity between 61 and 80) end bucket4,
       case when (select count(*)
                  from store_sales
                  where ss_quantity between 81 and 100) > 165306
            then (select avg(ss_ext_discount_amt)
                  from store_sales
                  where ss_quantity between 81 and 100)
            else (select avg(ss_net_paid)
                  from store_sales
                  where ss_quantity between 81 and 100) end bucket5
from reason
where r_reason_sk = 1;
""",
        """
select  
  cd_gender,
  cd_marital_status,
  cd_education_status,
  count(*) cnt1,
  cd_purchase_estimate,
  count(*) cnt2,
  cd_credit_rating,
  count(*) cnt3,
  cd_dep_count,
  count(*) cnt4,
  cd_dep_employed_count,
  count(*) cnt5,
  cd_dep_college_count,
  count(*) cnt6
 from
  customer c,customer_address ca,customer_demographics
 where
  c.c_current_addr_sk = ca.ca_address_sk and
  ca_county in ('Rush County','Toole County','Jefferson County','Dona Ana County','La Porte County') and
  cd_demo_sk = c.c_current_cdemo_sk and 
  exists (select *
          from store_sales,date_dim
          where c.c_customer_sk = ss_customer_sk and
                ss_sold_date_sk = d_date_sk and
                d_year = 2002 and
                d_moy between 1 and 1+3) and
   (exists (select *
            from web_sales,date_dim
            where c.c_customer_sk = ws_bill_customer_sk and
                  ws_sold_date_sk = d_date_sk and
                  d_year = 2002 and
                  d_moy between 1 ANd 1+3) or 
    exists (select * 
            from catalog_sales,date_dim
            where c.c_customer_sk = cs_ship_customer_sk and
                  cs_sold_date_sk = d_date_sk and
                  d_year = 2002 and
                  d_moy between 1 and 1+3))
 group by cd_gender,
          cd_marital_status,
          cd_education_status,
          cd_purchase_estimate,
          cd_credit_rating,
          cd_dep_count,
          cd_dep_employed_count,
          cd_dep_college_count
 order by cd_gender,
          cd_marital_status,
          cd_education_status,
          cd_purchase_estimate,
          cd_credit_rating,
          cd_dep_count,
          cd_dep_employed_count,
          cd_dep_college_count
 limit 5;
""",
    ]
    if r.button("Get query 1"):
        query1 = """
  select  a.ca_state state, count(*) cnt
 from customer_address a
     ,customer c
     ,store_sales s
     ,date_dim d
     ,item i
 where       a.ca_address_sk = c.c_current_addr_sk
 	and c.c_customer_sk = s.ss_customer_sk
 	and s.ss_sold_date_sk = d.d_date_sk
 	and s.ss_item_sk = i.i_item_sk
 	and d.d_month_seq = 
 	     (select distinct (d_month_seq)
 	      from date_dim
               where d_year = {0}
 	        and d_moy = {1} )
 	and i.i_current_price > 1.2 * 
             (select avg(j.i_current_price) 
 	     from item j 
 	     where j.i_category = i.i_category)
 group by a.ca_state
 having count(*) >= 10
 order by cnt 
  limit 100;
""".format(
            q1year, q1month
        )
        with r.expander("Show code"):
            r.code(query1)
        r.write("Query 1 Connected")
        r.write("Question: List all the states with at least 10 customers who during a given month bought items with the price tag at least 20% higher than the average price of items in the same category.",)
        connection = engine.connect()
        res = pd.read_sql(query1, connection)
        engine.dispose()
        r.dataframe(res)
    for query, table_name, question in zip(queries, tables, questions):
        connection = engine.connect()
        l.write(table_name + " Connected")
        l.write("Query: " + question)
        res = pd.read_sql(query, connection)
        engine.dispose()
        l.dataframe(res)
        l.write("Running Next Query")


def part2():
    # Snowpark initialization
    st.write(
        "<style>[data-testid='stMetricLabel'] {min-height: 0.5rem !important}</style>",
        unsafe_allow_html=True,
    )
    st.image(APP_ICON_URL, width=80)
    st.title("Predict Customer Lifetime Value Using XGBoost")
    session = create_session()
    session.sql_simplifier_enabled = True
    session.add_import("@ml_models/model.joblib")
    session.add_packages(
        "snowflake-snowpark-python",
        "scikit-learn",
        "pandas",
        "numpy",
        "joblib",
        "cachetools",
        "xgboost",
        "joblib",
    )

    # Define the list of features and their data types
    st.header("Input Features")
    features = [
        "C_BIRTH_YEAR",
        "CA_ZIP",
        "CD_GENDER",
        "CD_MARITAL_STATUS",
        "CD_CREDIT_RATING",
        "CD_EDUCATION_STATUS",
        "CD_DEP_COUNT",
    ]
    data_types = [int, str, str, str, str, str, int]

    # Define inputs
    inputs = []
    birth_year_range = (1960, 2020)
    gender_options = ["Male", "Female"]
    marital_status_options = ["Married", "Widowed", "Single", "Divorced"]
    credit_rating_options = ["Good", "High Risk", "Unknown"]
    education_status_options = [
        "Advanced Degree",
        "Unknown",
        "4 yr Degree",
        "Primary",
        "2 yr Degree",
        "Secondary",
        "College",
    ]
    pin_code_locations = ['24128','76232','65084','87816','83926','77556',
                          '20548','26231','43848','15126','91137',
                          '61265','98294','25782','17920','18426',
                          '98235','40081','84093','28577','55565',
                          '17183','54601','67897','22752','86284',
                          '18376','38607','45200','21756','29741',
                          '96765','23932','89360','29839','25989',
                          '28898','91068','72550','10390','18845',
                          '47770','82636','41367','76638','86198',
                          '81312','37126','39192','88424','72175',
                          '81426','53672','10445','42666','66864',
                          '66708','41248','48583','82276','18842',
                          '78890','49448','14089','38122','34425',
                          '79077','19849','43285','39861','66162',
                          '77610','13695','99543','83444','83041',
                          '12305','57665','68341','25003','57834',
                          '62878','49130','81096','18840','27700',
                          '23470','50412','21195','16021','76107',
                          '71954','68309','18119','98359','64544',
                          '10336','86379','27068','39736','98569',
                          '28915','24206','56529','57647','54917',
                          '42961','91110','63981','14922','36420',
                          '23006','67467','32754','30903','20260',
                          '31671','51798','72325','85816','68621',
                          '13955','36446','41766','68806','16725',
                          '15146','22744','35850','88086','51649',
                          '18270','52867','39972','96976','63792',
                          '11376','94898','13595','10516','90225',
                          '58943','39371','94945','28587','96576',
                          '57855','28488','26105','83933','25858',
                          '34322','44438','73171','30122','34102',
                          '22685','71256','78451','54364','13354',
                          '45375','40558','56458','28286','45266',
                          '47305','69399','83921','26233','11101',
                          '15371','69913','35942','15882','25631',
                          '24610','44165','99076','33786','70738',
                          '26653','14328','72305','62496','22152',
                          '10144','64147','48425','14663','21076',
                          '18799','30450','63089','81019','68893',
                          '24996','51200','51211','45692','92712',
                          '70466','79994','22437','25280','38935',
                          '71791','73134','56571','14060','19505',
                          '72425','56575','74351','68786','51650',
                          '20004','18383','76614','11634','18906',
                          '15765','41368','73241','76698','78567',
                          '97189','28545','76231','75691','22246',
                          '51061','90578','56691','68014','51103',
                          '94167','57047','14867','73520','15734',
                          '63435','25733','35474','24676','94627',
                          '53535','17879','15559','53268','59166',
                          '11928','59402','33282','45721','43933',
                          '68101','33515','36634','71286','19736',
                          '58058','55253','67473','41918','19515',
                          '36495','19430','22351','77191','91393',
                          '49156','50298','87501','18652','53179',
                          '18767','63193','23968','65164','68880',
                          '21286','72823','58470','67301','13394',
                          '31016','70372','67030','40604','24317',
                          '45748','39127','26065','77721','31029',
                          '31880','60576','24671','45549','13376',
                          '50016','33123','19769','22927','97789',
                          '46081','72151','15723','46136','51949',
                          '68100','96888','64528','14171','79777',
                          '28709','11489','25103','32213','78668',
                          '22245','15798','27156','37930','62971',
                          '21337','51622','67853','10567','38415',
                          '15455','58263','42029','60279','37125',
                          '56240','88190','50308','26859','64457',
                          '89091','82136','62377','36233','63837',
                          '58078','17043','30010','60099','28810',
                          '98025','29178','87343','73273','30469',
                          '64034','39516','86057','21309','90257',
                          '67875','40162','11356','73650','61810',
                          '72013','30431','22461','19512','13375',
                          '55307','30625','83849','68908','26689',
                          '96451','38193','46820','88885','84935',
                          '69035','83144','47537','56616','94983',
                          '48033','69952','25486','61547','27385',
                          '61860','58048','56910','16807','17871',
                          '35258','31387','35458','35576']

    # User prompts for input
    inp, out = st.columns(2)
    for i, feature in enumerate(features):
        if data_types[i] == int:
            if feature == "C_BIRTH_YEAR":
                inputs.append(
                    int(
                        inp.slider(
                            f"Enter {feature}",
                            min_value=birth_year_range[0],
                            max_value=birth_year_range[1],
                        )
                    )
                )
            else:
                dep_count = inp.selectbox(f"Enter {feature}", options=["0", "1"])
                if dep_count == "1":
                    inputs.append(1)
                else:
                    inputs.append(0)
        else:
            if feature == "CA_ZIP":
                inputs.append(
                    inp.selectbox(f"Enter {feature}", options=pin_code_locations)
                )
            elif feature == "CD_GENDER":
                gender_input = inp.selectbox(f"Enter {feature}", options=gender_options)
                if gender_input == "Male":
                    inputs.append("M")
                else:
                    inputs.append("F")
            elif feature == "CD_MARITAL_STATUS":
                marital_status = inp.selectbox(
                    f"Enter {feature}", options=marital_status_options
                )
                if gender_input == "Married":
                    inputs.append("M")
                elif gender_input == "Widowed":
                    inputs.append("W")
                elif gender_input == "Single":
                    inputs.append("S")
                else:
                    inputs.append("D")

            elif feature == "CD_CREDIT_RATING":
                inputs.append(
                    inp.selectbox(f"Enter {feature}", options=credit_rating_options)
                )
            else:
                inputs.append(
                    inp.selectbox(f"Enter {feature}", options=education_status_options)
                )

    # Show user inputs as DF
    df_inputs = session.create_dataframe([inputs], schema=features)
    out.dataframe(df_inputs)
    st.columns([2, 1])
    # Call predict function on button click
    if st.button("Predict CLV"):
        out.write("Intializing SnowPark...")
        out.subheader("MAPE - 10.486622 % | Model Accuracy - 90.52 %")
        df_inputs = session.create_dataframe([inputs], schema=features)

        # Add util funcs to call UDF on user input
        @cachetools.cached(cache={})
        def read_file(filename):

            import_dir = sys._xoptions.get("snowflake_import_directory")
            if import_dir:
                with open(os.path.join(import_dir, filename), "rb") as file:
                    m = joblib.load(file)
                    return m

        @F.pandas_udf(
            session=session,
            max_batch_size=10000,
            is_permanent=True,
            stage_location="@ml_models",
            replace=True,
            name="clv_xgboost_udf",
        )
        def predict(
            df: T.PandasDataFrame[int, str, str, str, str, str, int]
        ) -> T.PandasSeries[float]:
            m = read_file("model.joblib")
            df.columns = features
            return m.predict(df)

        @st.cache_data(show_spinner=False)
        def predictCLV(_df):
            df_predicted_clv = df_inputs.select(*df_inputs, predict(*df_inputs).alias("PREDICTION"))
            df_predicted_clv = pd.DataFrame(df_predicted_clv.collect())
            return df_predicted_clv

        out.write("Intialized")

        out.write("Calling Snowpark XGBoost UDF on user given input ")
        df_predicted_clv = predictCLV(df_inputs)
        out.header("Predicted CLV for User Given Inputs")
        out.dataframe(df_predicted_clv)
        out.header(
            f"This customer is expected to invest a total of ${'{:,}'.format(int(df_predicted_clv.iloc[0]['PREDICTION']) )} in our company"
        )


with st.container():
    st.header("TCPDS Queries")
    part1()
with st.container():
    part2()

