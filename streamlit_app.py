# Imports
import streamlit as st
import pandas as pd
import plotly.express as px
import great_expectations as gx
import io
st.title("Assignment 1")
st.subheader("EDA on Hufty Bikes")
# Data Validation
# 1. Df info
dftr = gx.read_excel("KPMG_VI_New_raw_data_update_final.xlsx",
                     sheet_name="Transactions", skiprows=1)
dfca = gx.read_excel("KPMG_VI_New_raw_data_update_final.xlsx",
                     sheet_name="CustomerAddress", skiprows=1)
dfcd = gx.read_excel("KPMG_VI_New_raw_data_update_final.xlsx",
                     sheet_name="CustomerDemographic")
dfnc = gx.read_excel("./KPMG_VI_New_raw_data_update_final.xlsx",
                     sheet_name="NewCustomerList", skiprows=1)

st.subheader("Dataset Information")
df_trad = pd.merge(dftr, dfca, on='customer_id', how='outer')
df = pd.merge(df_trad, dfcd, on='customer_id', how='outer')
st.write("Unique Transaction IDs - ", len(df))
# df.dropna(inplace=True)
# a = pd.DataFrame(df.describe())
# st.dataframe(a)

# 2. GE validations
st.subheader("Data validation with great expectation")

tid_val = dftr.expect_column_values_to_be_unique("transaction_id")
tdate_val = dftr.expect_column_values_to_be_of_type(
    column="transaction_date", type_="datetime64")
torders_val = dftr.expect_column_values_to_not_be_null(
    column=['order_status', 'brand'])
tprice = dftr.expect_column_values_to_be_between(
    column="list_price", min_value=10, max_value=2100)

st.write("1. Unique IDs check on transaction_id")
st.write("Unexpected_count :", tid_val.result["unexpected_count"])

st.write("2. Dates validations")
st.write(tdate_val.result)

st.write("3. Null values check on order_status and brand")
st.write("Unexpected_count :", torders_val.result["unexpected_count"])

st.write("4. Prices are relavant and in between a tight range of 10-2100 $USD")
st.write("Unexpected_count :", tprice.result["unexpected_count"])

# Data Analysis

# 1. Brand - Units sold
st.subheader("1. Brand - Units sold")
fig1 = px.histogram(df, x="brand", y="transaction_id",
                    labels=dict(x="Number of customers"),
                    height=500,
                    histfunc="count"
                    )
st.plotly_chart(fig1, use_container_width=True)

# 2. Brand - Profit generated
st.subheader("2. Brand - Profits Contributed in $USD")
df["profit"] = df["list_price"] - df["standard_cost"]
profit_sum = pd.DataFrame(df.groupby("brand").sum()["profit"])
fig2 = px.pie(df, names="brand",
              values="profit",
              height=500,
              hole=0.3,
              )
st.plotly_chart(fig2, use_container_width=True)
st.dataframe(profit_sum)
st.write("Total profit generated - ",
         format(int(profit_sum["profit"].sum()), ","), "$")
st.write("Insight - Although the units sold by the brands are almost equal, Solex and WeAreA2B together contribute 47% of the total profit generated.")

# 3. Customer - Demographic attributes that do not effect sales
st.subheader("3. Customer attributes - Revenue")
st.write("3.1 Owning cars")
st.plotly_chart(px.pie(df, names="owns_car", values="list_price", height=400))
st.write("3.2 Gender")
st.plotly_chart(px.pie(df, names="gender", values="list_price", height=400))
st.write("3.3 Online / Offline channel")
st.plotly_chart(px.pie(df, names="online_order",
                values="list_price", height=400))
st.write("Insights - These customer attributes do not affect sales. Gender, Online/Offline Channel, Owning cars")

# 4. Customer - Demographic attributes that effect sales
st.subheader("4. Customer location analysis")
profit_old = pd.DataFrame(
    df.groupby("state").sum()["past_3_years_bike_related_purchases"])
# st.dataframe(profit_old)
st.plotly_chart(px.pie(profit_old, title="Existing customer data", names=[
                "NSW", "QLD", "VIC", "Victoria", "New South Wales"], values="past_3_years_bike_related_purchases"))

profit_new = pd.DataFrame(
    dfnc.groupby("state").sum()["past_3_years_bike_related_purchases"])
# st.dataframe(profit_new)
st.plotly_chart(px.pie(profit_new, names=[
                "NSW", "QLD", "VIC"], values="past_3_years_bike_related_purchases", title="New customer data"))
st.write("Insight - New customers are mostly in the same region as the old customers validating that the locations are targeted accurately for the buisiness")

st.subheader("5. Customer class & Product class sales analysis")
st.write("5.1 Wealth segment")
profit_job_title = pd.DataFrame(
    df.groupby("wealth_segment").sum()["list_price"])
st.dataframe(profit_job_title)
st.plotly_chart(px.pie(profit_job_title, names=[
                "Affluent Customer", "High Net Worth", "Mass Customer"], values="list_price"))

st.write("5.2 Product line sales")
fig6 = px.histogram(df, x="product_line", y="transaction_id",
                    labels=dict(x="Number of customers"),
                    height=500,
                    histfunc="count"
                    )
st.plotly_chart(fig6, use_container_width=True)

st.write("5.3 Customer wealth (based on residance area)")
fig7 = px.histogram(df, y='customer_id',
                    x='property_valuation', histfunc="count")
st.plotly_chart(fig7, use_container_width=True)

st.write("5.4 Customer age")
bins = [0, 18, 35, 48, 60,100]
labels = ['Teen (0-18)', 'Young Adults (18-35)', 'Adults (35-48)', 'Adults (48-60)', 'Old (60+)']
df['age_ranges'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

fig7 = px.histogram(df, y='transaction_id',
                    x='age_ranges', histfunc="count")
st.plotly_chart(fig7, use_container_width=True)

profit_age_range = pd.DataFrame(
    df.groupby("product_class").sum()["profit"])
st.dataframe(profit_age_range)
fig8 = px.pie(profit_age_range, names=['high','low','medium'], values='profit',title="Product class revenue")
st.plotly_chart(fig8, use_container_width=True)
