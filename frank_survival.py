import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, statistics
from lifelines import CoxPHFitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from sklearn.preprocessing import LabelEncoder
from io import StringIO
import os
import json
import numpy as np
import plotly.express as px





# Initialize LLM
llm = ChatOpenAI(temperature=0, api_key=OPEN_AI_KEY)

def infer_plotting_intent(query):
    messages = [
        SystemMessage(content="""
            You are an expert data analyst. Your task is to decide if the user query relates to:
            1. Cox hazard/log-rank survival analysis (answer: 'survival')
            2. General plotting like bar charts, line charts, scatter plots (answer: 'plot')
            Answer only with 'survival' or 'plot'.
        """),
        HumanMessage(content=query)
    ]
    response = llm(messages).content.strip().lower()
    return response

def infer_columns(query, df_columns):
    messages = [
        SystemMessage(content=f"""
            You are a helpful assistant. Given the following columns: {df_columns},
            and the user query, identify relevant columns (as a Python list of strings).
            Just return the list of column names in Python syntax.
        """),
        HumanMessage(content=query)
    ]
    response = llm(messages).content.strip()
    try:
        inferred = eval(response)
        if isinstance(inferred, list):
            return inferred
    except:
        pass
    return []

def ask_llm_to_infer_column(df):
    '''
    cols = ask_llm_about_survival_curve(df)
    required_cols = ['duration', 'event'] + ([cols['group']] if 'group' in cols else [])
    df = df[[cols['duration'], cols['event']] + ([cols['group']] if 'group' in cols else [])]
    df.columns = ['duration', 'event'] + (['group'] if 'group' in cols else [])

    '''
    prompt = f"""
    I have a dataset with the following columns: {', '.join(df.columns)}.
    I want to generate survival curves.
    Which columns should be used for duration and event?
    {{"duration": "col1", "event": "col2"}}
    """

    chat = ChatOpenAI(model_name="gpt-4-0613",api_key=OPEN_AI_KEY)
    messages = [
        SystemMessage(content="You are a helpful assistant that identifies survival analysis columns."),
        HumanMessage(content=prompt)
    ]
    response = chat(messages)

    return json.loads(response.content)


def ask_llm_about_survival_curve(df):
    '''
    cols = ask_llm_about_survival_curve(df)
    required_cols = ['duration', 'event'] + ([cols['group']] if 'group' in cols else [])
    df = df[[cols['duration'], cols['event']] + ([cols['group']] if 'group' in cols else [])]
    df.columns = ['duration', 'event'] + (['group'] if 'group' in cols else [])

    '''
    prompt = f"""
    I have a dataset with the following columns: {', '.join(df.columns)}.
    I want to generate survival curves.
    Which columns should be used for duration and event? Optionally, identify a group column if available.
    Return your answer in a JSON format like:
    {{"duration": "col1", "event": "col2", "group": "col3"}} or {{"duration": "col1", "event": "col2"}}
    """

    chat = ChatOpenAI(model_name="gpt-4-0613",api_key=OPEN_AI_KEY)
    messages = [
        SystemMessage(content="You are a helpful assistant that identifies survival analysis columns."),
        HumanMessage(content=prompt)
    ]
    response = chat(messages)

    return json.loads(response.content)


def infer_covariates(query, df_columns):
    messages = [
        SystemMessage(content=f"""
            You are a helpful assistant. Given the following columns: {df_columns},
            and the user query, identify covariates for a Cox proportional hazards model (as a Python list of strings).
            Just return the list of covariate column names in Python syntax. Do not include the duration, event, or group column.
        """),
        HumanMessage(content=query)
    ]
    response = llm(messages).content.strip()
    try:
        inferred = eval(response)
        if isinstance(inferred, list):
            return inferred
    except:
        pass
    return []

def generate_km_curve(df):
    kmf = KaplanMeierFitter()
    fig = px.line()

    if 'group' in df.columns:
        for name, grouped_df in df.groupby("group"):
            kmf.fit(grouped_df["duration"], grouped_df["event"], label=f"KM - {name}")
            survival_df = kmf.survival_function_.reset_index()
            fig.add_scatter(x=survival_df["timeline"], y=survival_df[kmf._label], name=f"KM - {name}")
    else:
        kmf.fit(df["duration"], df["event"], label="KM - Overall")
        survival_df = kmf.survival_function_.reset_index()
        fig.add_scatter(x=survival_df["timeline"], y=survival_df[kmf._label], name="KM - Overall")

    fig.update_layout(title="Kaplan-Meier Survival Curves",
                      xaxis_title="Time",
                      yaxis_title="Survival Probability")
    return fig


def generate_cox_fitting(df):
    cph = CoxPHFitter()
    try:
        df_cox = df.copy()
        df_cox = pd.get_dummies(df_cox, drop_first=True)
        cph.fit(df_cox, duration_col="duration", event_col="event")
        fig = cph.plot()
        #return cph.summary
        return fig
    except Exception as e:
        return f"Cox model failed: {e}"

def main():
    st.title("LLM-Powered Plotting Tool")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    user_query = st.text_input("Enter your plot request")

    if uploaded_file and user_query:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.write(df.head())

        intent = infer_plotting_intent(user_query)
        cols = ask_llm_to_infer_column(df)
        
        required_cols = ['duration', 'event'] + ([cols['group']] if 'group' in cols else [])

        data_df = df[[cols['duration'], cols['event']]] #+ ([cols['group']] if 'group' in cols else [])]
        data_df.columns = ['duration', 'event'] #+ (['group'] if 'group' in cols else [])
        
        st.subheader("Kaplan-Meier Curve")
        st.plotly_chart(generate_km_curve(data_df), use_container_width=True)

        #st.subheader("Cox Fitting Curve")
        #st.plotly_chart(generate_cox_fitting(data_df), use_container_width=True)
        cph = CoxPHFitter()
        df_cox = df.copy()
        inferred_column_dict = {v: k for k, v in cols.items()}
        df_cox = df_cox.rename(columns=inferred_column_dict)
        #df_cox = pd.get_dummies(df_cox, drop_first=True)
        df_cox = df_cox.apply(LabelEncoder().fit_transform)
        cph.fit(df_cox, duration_col="duration", event_col="event")
        st.pyplot(cph.plot().figure)
        #st.plotly_chart(fig, use_container_width=True)

        # Covariates
        covariates = infer_covariates(user_query, df_cox.columns)
        
        cova = covariates[-1]
       
        fig2, ax2 = plt.subplots()
        bins = np.linspace(df_cox[cova].min(), 
                           df_cox[cova].max(), 5)
        
        cph.plot_partial_effects_on_outcome(covariates=cova, values=bins.tolist()  , ax=ax2)
        st.pyplot(fig2.figure) 
      


if __name__ == "__main__":
    main()
