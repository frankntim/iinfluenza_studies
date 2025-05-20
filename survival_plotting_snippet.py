import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from lifelines import KaplanMeierFitter, NelsonAalenFitter, BreslowFlemingHarringtonFitter, CoxPHFitter,  WeibullFitter
#from langchain.chat_models import ChatOpenAI
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
import json
import tempfile
import os
import re




OPEN_AI_KEY = ""


# LangChain setup with memory
llm = ChatOpenAI(model_name="gpt-4-0613", api_key=OPEN_AI_KEY)
memory = ConversationBufferMemory(memory_key="chat_history")


def survival_plotting_agent(df: pd.DataFrame):
    def identify_survival_columns(df: pd.DataFrame) -> dict:
        prompt = f"""
        I have a dataset with the following columns: {', '.join(df.columns)}.
        I want to generate survival curves.
        Which columns should be used for duration and event? Optionally, identify a group column if available.
        Return JSON like: {{"duration": "col1", "event": "col2", "group": "col3"}} or {{"duration": "col1", "event": "col2"}}
        """
        messages = [
            SystemMessage(content="You are a helpful assistant that identifies survival analysis columns."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        memory.chat_memory.add_user_message(prompt)
        memory.chat_memory.add_ai_message(response.content)
        return json.loads(response.content)
    
    def generate_survival_plots(df: pd.DataFrame, columns: dict) -> dict:
        output = {}
        df = df[[columns['duration'], columns['event']] + ([columns['group']] if 'group' in columns else [])]
        df.columns = ['duration', 'event'] + (['group'] if 'group' in columns else [])

        def curve_plot(fitter_cls, title, y_label):
            fitter = fitter_cls()
            fig = px.line()
            if 'group' in df.columns:
                for name, grouped_df in df.groupby("group"):
                    fitter.fit(grouped_df["duration"], grouped_df["event"], label=f"{title} - {name}")
                    curve_df = (fitter.survival_function_ if hasattr(fitter, 'survival_function_') else fitter.cumulative_hazard_).reset_index()
                    fig.add_scatter(x=curve_df["timeline"], y=curve_df[fitter._label], name=fitter._label)
            else:
                fitter.fit(df["duration"], df["event"], label=f"{title} - Overall")
                curve_df = (fitter.survival_function_ if hasattr(fitter, 'survival_function_') else fitter.cumulative_hazard_).reset_index()
                fig.add_scatter(x=curve_df["timeline"], y=curve_df[fitter._label], name=fitter._label)
            fig.update_layout(title=f"{title} Curve", xaxis_title="Time", yaxis_title=y_label)
            return fig.to_dict()

        output['Kaplan-Meier'] = curve_plot(KaplanMeierFitter, "KM", "Survival Probability")
        output['Nelson-Aalen'] = curve_plot(NelsonAalenFitter, "NA", "Cumulative Hazard")
        output['Breslow-Fleming-Harrington'] = curve_plot(BreslowFlemingHarringtonFitter, "BFH", "Survival Probability")
        return output
    
    def generate_combined_survival_plot(df: pd.DataFrame, columns: dict):
        df = df[[columns['duration'], columns['event']] + ([columns['group']] if 'group' in columns else [])]
        df.columns = ['duration', 'event'] + (['group'] if 'group' in columns else [])

        fitters = [
            (KaplanMeierFitter, "Kaplan-Meier"),
            (NelsonAalenFitter, "Nelson-Aalen"),
            #(WeibullFitter, "Weibull")
        ]

        fig = go.Figure()

        for fitter_cls, title in fitters:
            fitter = fitter_cls()
            if 'group' in df.columns:
                for name, grouped_df in df.groupby("group"):
                    fitter.fit(grouped_df["duration"], grouped_df["event"], label=f"{title} - {name}")
                    curve_df = (fitter.survival_function_ if hasattr(fitter, 'survival_function_') else fitter.cumulative_hazard_).reset_index()
                    fig.add_trace(go.Scatter(x=curve_df["timeline"], y=curve_df[fitter._label], mode='lines', name=fitter._label))
            else:
                fitter.fit(df["duration"], df["event"], label=f"{title} - Overall")
                curve_df = (fitter.survival_function_ if hasattr(fitter, 'survival_function_') else fitter.cumulative_hazard_).reset_index()
                fig.add_trace(go.Scatter(x=curve_df["timeline"], y=curve_df[fitter._label], mode='lines', name=fitter._label))

        fig.update_layout(
            title="Combined Survival Curves",
            xaxis_title="Timeline",
            yaxis_title="Survival Probability / Cumulative Hazard",
            legend_title="Model"
        )
        return fig
    
    def generate_survival_subplots(df: pd.DataFrame, columns: dict):
        df = df[[columns['duration'], columns['event']] + ([columns['group']] if 'group' in columns else [])]
        df.columns = ['duration', 'event'] + (['group'] if 'group' in columns else [])

        fitters = [
            (KaplanMeierFitter, "Kaplan-Meier"),
            (NelsonAalenFitter, "Nelson-Aalen"),
            (BreslowFlemingHarringtonFitter, "BH")
        ]

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=[name for _, name in fitters])

        for i, (fitter_cls, title) in enumerate(fitters, start=1):
            fitter = fitter_cls()
            if 'group' in df.columns:
                for name, grouped_df in df.groupby("group"):
                    fitter.fit(grouped_df["duration"], grouped_df["event"], label=f"{title} - {name}")
                    curve_df = (fitter.survival_function_ if hasattr(fitter, 'survival_function_') else fitter.cumulative_hazard_).reset_index()
                    fig.add_trace(
                        go.Scatter(x=curve_df["timeline"], y=curve_df[fitter._label], mode='lines', name=fitter._label),
                        row=i, col=1
                    )
            else:
                fitter.fit(df["duration"], df["event"], label=f"{title} - Overall")
                curve_df = (fitter.survival_function_ if hasattr(fitter, 'survival_function_') else fitter.cumulative_hazard_).reset_index()
                fig.add_trace(
                    go.Scatter(x=curve_df["timeline"], y=curve_df[fitter._label], mode='lines', name=fitter._label),
                    row=i, col=1
                )

        fig.update_layout(
            height=900,
            title_text="Survival Curves by Method",
            showlegend=True
        )
        return fig
    

    col_ids = identify_survival_columns(df)
    survival_plots = generate_survival_plots(df, col_ids)
    fig = generate_survival_subplots(df, col_ids)

    return fig


def general_plotting_agent(df: pd.DataFrame, user_query: str) -> list:
    def get_fig_from_code(code):
        local_variables = {}
        exec(code, {}, local_variables)
        return local_variables['fig']

    CHART_PROMPT = '''
    Role: You are biomarker assistant tasked with providing up-to-date information about the clinical trial data.

    Objective: Your job is to write pandas code to visualize and plot charts based on the given data and column names, context and task.

    Capabilities: You have been given a dataframe as data. The data columns are provided. The context of the task is presented as context.

    Next Steps: When asked to visualize data, provide only python code using Plotly Express to generate charts. 
    The plotting figure sizes should be 5 by 3. Select different colours and styles.
    Include all necessary imports in your code snippets.
    If the task request specific graph use it otherwise use appropriate chart
    

    columns: {columns}

    data: {data}

    context: {context}

    prompt: {task}

    '''
    result_description = ""
    chart_prompt = ChatPromptTemplate.from_template(CHART_PROMPT)
    graph_agent_chain = chart_prompt | llm

    graph_agent_response = graph_agent_chain.invoke({
                "columns":df.columns.tolist(),
                "data": df.to_dict(orient="records"),
                "context": result_description,
                "task": user_query
            })
    
    
    graph_result_output = graph_agent_response.content.strip() or ""
    code_block_match = re.search(r'```(?:[Pp]ython)?(.*?)```', graph_result_output, re.DOTALL)
    if code_block_match:
        
        code_block = code_block_match.group(1).strip()
        cleaned_code = re.sub(r'(?m)^\s*fig\.show\(\)\s*$', '', code_block)
        try:
            fig = get_fig_from_code(cleaned_code)
            return cleaned_code, fig
        except:
            fig = ""
            cleaned_code = ""
            pass
    else:
        fig = ""
        cleaned_code = ""
        return cleaned_code,fig
    
# Multi-agent decision routing
def route_query(user_query: str) -> str:
    messages = [
        SystemMessage(content="You are a routing assistant. Decide if the query is about survival analysis or general plotting. Return 'survival' or 'general' only."),
        HumanMessage(content=f"Query: {user_query}")
    ]
    response = llm.invoke(messages)
    return response.content.strip().lower()




uploaded_file = st.file_uploader("Upload your CSV survival data file")
user_query = st.text_input("What would you like to visualize?")

if uploaded_file is not None and user_query:
   
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    task = route_query(user_query)
    with st.spinner(f"Routing to {task} agent..."):
        if task == "survival":
            fig = survival_plotting_agent(df)
            st.plotly_chart(fig, use_container_width=True)
        elif task == "general":
            cleaned_code, fig = general_plotting_agent(df,user_query)
            st.plotly_chart(fig, use_container_width=True)
            st.write(cleaned_code)
        else:
            st.warning("Could not generate plot.")

    #survival_plots = survival_plotting_agent(df)
    #fig = survival_plotting_agent(df)
    #st.plotly_chart(fig, use_container_width=True)
    #st.write(pio.to_code(fig))
    #st.write(fig.to_json())
    #for name, fig_dict in survival_plots.items():
    #    fig = go.Figure(fig_dict)
    #    st.plotly_chart(fig, use_container_width=True)


    #cleaned_code, fig = general_plotting_agent(df,user_query)
    #if fig:
    #    st.plotly_chart(fig, use_container_width=True)
    #else:
        #st.warning("Could not generate plot.")

    # Create a react agent with all the functions as tools