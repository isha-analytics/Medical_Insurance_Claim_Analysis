# -----------------------------
# IMPORT LIBRARIES
# -----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Insurance Claim Dashboard",layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
st.sidebar.title("Dataset")

default_path = r"Insurance claim Analysis.csv"

try:
    df = pd.read_csv(default_path)
    st.sidebar.success("Default dataset loaded")
except:
    uploaded_file = st.sidebar.file_uploader("Upload Dataset")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Upload dataset to continue")
        st.stop()

# -----------------------------
# DATA CLEANING
# -----------------------------
df['age'] = pd.to_numeric(df['age'],errors='coerce')

bins =[0,25,35,45,55,65,100]
labels=['18-25','26-35','36-45','46-55','56-65','65+']
df['age_group']=pd.cut(df['age'],bins=bins,labels=labels)

df['claim_date']=pd.to_datetime(df['claim_date'])
df['month']=df['claim_date'].dt.month_name()
df['year']=df['claim_date'].dt.year


# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.title("Filters")

gender = st.sidebar.multiselect("Gender",df['gender'].unique(),default=df['gender'].unique())
policy = st.sidebar.multiselect("Policy Type",df['policy_type'].unique(),default=df['policy_type'].unique())
city = st.sidebar.multiselect("City",df['city'].unique(),default=df['city'].unique())

filtered_df = df[(df['gender'].isin(gender)) &
                 (df['policy_type'].isin(policy)) &
                 (df['city'].isin(city))]

# -----------------------------
# PAGE NAVIGATION
# -----------------------------
page = st.sidebar.radio("Select Page",
["Summary Statistics","KPI Dashboard","Visualizations","Prediction"])

# -----------------------------
# SUMMARY PAGE
# -----------------------------
if page=="Summary Statistics":

    st.title("Insurance Claim Dataset Overview")

    st.subheader("Dataset Shape")
    st.write(filtered_df.shape)

    st.subheader("Column Names")
    st.write(filtered_df.columns)

    st.subheader("Data Types")
    st.write(filtered_df.dtypes)

    st.subheader("Missing Values")
    st.write(filtered_df.isnull().sum())

    st.subheader("Duplicate Rows")
    st.write(filtered_df.duplicated().sum())

    st.subheader("Statistical Summary")
    st.write(filtered_df.describe())

    st.subheader("Preview Data")
    st.dataframe(filtered_df.head())


# -----------------------------
# KPI PAGE
# -----------------------------
elif page=="KPI Dashboard":

    st.title("Key Performance Indicators")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Total Claims",len(filtered_df))
    col2.metric("Total Claim Amount",int(filtered_df['total_claim_amount'].sum()))
    col3.metric("Average Claim Amount",int(filtered_df['total_claim_amount'].mean()))
    col4.metric("Max Claim Amount",int(filtered_df['total_claim_amount'].max()))

    st.subheader("Claims by Gender")
    fig = px.pie(filtered_df,names="gender")
    st.plotly_chart(fig,use_container_width=True)


# -----------------------------
# VISUALIZATION PAGE
# -----------------------------
elif page=="Visualizations":

    st.title("Insurance Claim Analysis")

# 1 Age group claims
    age_claim=filtered_df.groupby('age_group')['total_claim_amount'].sum().reset_index()
    fig=px.bar(age_claim,x='age_group',y='total_claim_amount',title="1. Claim Amount by Age Group")
    st.plotly_chart(fig,use_container_width=True)

# 2 Hospital highest claim
    hospital_claim=filtered_df.groupby('hospital_name')['total_claim_amount'].sum().reset_index().sort_values(by='total_claim_amount',ascending=False).head(10)
    fig=px.bar(hospital_claim,x='hospital_name',y='total_claim_amount',title="2. Top Hospitals by Claim Amount")
    st.plotly_chart(fig)

# 3 Occupation claims
    occ=filtered_df['occupation_type'].value_counts().reset_index()
    occ.columns=['occupation_type','count']
    fig=px.bar(occ,x='occupation_type',y='count',title="3. Claims by Occupation")
    st.plotly_chart(fig)

# 4 Gender claims
    gender_count=filtered_df['gender'].value_counts().reset_index()
    gender_count.columns=['gender','count']
    fig=px.bar(gender_count,x='gender',y='count',title="4. Claims by Gender")
    st.plotly_chart(fig)

# 5 City claims
    city_claim=filtered_df['city'].value_counts().head(10).reset_index()
    city_claim.columns=['city','claims']
    fig=px.bar(city_claim,x='city',y='claims',title="5. Top Cities by Claims")
    st.plotly_chart(fig)

# 6 Policy type
    policy_claim=filtered_df['policy_type'].value_counts().reset_index()
    policy_claim.columns=['policy_type','claims']
    fig=px.bar(policy_claim,x='policy_type',y='claims',title="6. Claims by Policy Type")
    st.plotly_chart(fig)

# 7 Monthly claim amount
    monthly=filtered_df.groupby('month')['total_claim_amount'].sum().reset_index()
    fig=px.line(monthly,x='month',y='total_claim_amount',title="7. Monthly Claim Amount",markers=True)
    st.plotly_chart(fig)

# 8 Diet type
    diet=filtered_df['diet_type'].value_counts().reset_index()
    diet.columns=['diet_type','count']
    fig=px.pie(diet,names='diet_type',values='count',title="8. Claims by Diet Type")
    st.plotly_chart(fig)

# 9 ICU vs stay
    fig=px.scatter(filtered_df,x='icu_days',y='length_of_stay',title="9. ICU Days vs Length of Stay")
    st.plotly_chart(fig)

# 10 Claims by year
    year_claim=filtered_df.groupby('year').size().reset_index(name='claims')
    fig=px.line(year_claim,x='year',y='claims',markers=True,title="10. Claims by Year")
    st.plotly_chart(fig)

# 11 Agent channel
    agent=filtered_df['agent_channel'].value_counts().reset_index()
    agent.columns=['agent_channel','claims']
    fig=px.bar(agent,x='agent_channel',y='claims',title="11. Claims by Agent Channel")
    st.plotly_chart(fig)

# 12 Physical activity
    act=filtered_df['physical_activity_level'].value_counts().reset_index()
    act.columns=['activity','claims']
    fig=px.bar(act,x='activity',y='claims',title="12. Claims by Physical Activity Level")
    st.plotly_chart(fig)

# 13 Hospital tier
    tier=filtered_df.groupby('hospital_tier')['total_claim_amount'].sum().reset_index()
    fig=px.bar(tier,x='hospital_tier',y='total_claim_amount',title="13. Claim Amount by Hospital Tier")
    st.plotly_chart(fig)

# 14 Stay vs claim
    fig=px.scatter(filtered_df,x='length_of_stay',y='total_claim_amount',title="14. Stay vs Claim Amount")
    st.plotly_chart(fig)

# 15 Claim distribution
    fig=px.histogram(filtered_df,x='total_claim_amount',nbins=30,title="15. Distribution of Claim Amount")
    st.plotly_chart(fig)


# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page=="Prediction":

    st.title("Predict Insurance Claim Amount")

    features = ['age','icu_days','length_of_stay','hospital_bill_amount']
    df_model = filtered_df[features + ['total_claim_amount']].dropna()

    X = df_model[features]
    y = df_model['total_claim_amount']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    model = LinearRegression()
    model.fit(X_train,y_train)

    pred=model.predict(X_test)

    st.write("Model Accuracy (R2 Score):",round(r2_score(y_test,pred),2))

    st.subheader("Enter Patient Details")

    age=st.number_input("Age",20,90)
    icu=st.number_input("ICU Days",0,30)
    stay=st.number_input("Length of Stay",1,60)
    bill=st.number_input("Hospital Bill Amount",1000,1000000)

    if st.button("Predict Claim"):
        input_data=np.array([[age,icu,stay,bill]])
        prediction=model.predict(input_data)
        st.success(f"Predicted Claim Amount: ₹{int(prediction[0])}")
