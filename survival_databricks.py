import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from typing import Dict, Any, List, Tuple
import json
import numpy as np

from databricks.sdk import WorkspaceClient
from langchain_databricks import ChatDatabricks
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# --- Configuration ---
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID")
MODEL_ENDPOINT = os.getenv("DATABRICKS_MODEL_ENDPOINT", "databricks-dbrx-instruct")

if not all([DATABRICKS_HOST, DATABRICKS_TOKEN, GENIE_SPACE_ID]):
    st.error("Set DATABRICKS_HOST, DATABRICKS_TOKEN, GENIE_SPACE_ID env vars.")
    st.stop()

# --- Initialize WorkspaceClient (Genie via client.genie) ---
w = WorkspaceClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)

# --- LLM (Unchanged) ---
llm = ChatDatabricks(
    endpoint=MODEL_ENDPOINT,
    temperature=0.1,
    max_tokens=1500
)

# --- Tool: Query Genie (Using w.genie Directly) ---
@tool
def query_databricks_genie(natural_language_query: str) -> Dict[str, Any]:
    """Query Genie and return data as list of dicts + SQL."""
    try:
        with st.spinner(f"Genie: {natural_language_query}"):
            # Use w.genie.start_conversation_and_wait
            msg = w.genie.start_conversation_and_wait(
                space_id=GENIE_SPACE_ID,
                content=natural_language_query
            )
        if not msg.attachments:
            return {"error": "No data returned."}

        att = msg.attachments[0]
        # Use w.genie.get_message_attachment_query_result
        result = w.genie.get_message_attachment_query_result(
            space_id=GENIE_SPACE_ID,
            conversation_id=msg.conversation_id,
            message_id=msg.message_id,
            attachment_id=att.attachment_id
        )

        df = pd.DataFrame(result.rows, columns=result.columns) if result.rows else pd.DataFrame()
        return {
            "data": df.to_dict(orient="records"),
            "sql": att.sql,
            "columns": list(df.columns),
            "row_count": len(df)
        }
    except Exception as e:
        return {"error": str(e)}

# --- Plot: Kaplan-Meier (Unchanged) ---
def plot_kaplan_meier_plotly(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str = None
) -> Tuple[go.Figure, str]:
    """Return Plotly KM curve + summary text."""
    kmf = KaplanMeierFitter()
    fig = go.Figure()

    summary_lines = []
    p_value = None

    if group_col and group_col in df.columns:
        groups = df[group_col].dropna().unique()
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, g in enumerate(groups):
            mask = df[group_col] == g
            gdf = df[mask]
            if len(gdf) == 0:
                continue

            kmf.fit(gdf[time_col], gdf[event_col], label=str(g))
            sf = kmf.survival_function_
            ci = kmf.confidence_interval_

            fig.add_trace(go.Scatter(
                x=sf.index, y=sf.iloc[:, 0],
                mode='lines',
                name=str(g),
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f"<b>{g}</b><br>Time: %{{x}}<br>Survival: %{{y:.3f}}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=ci.index, y=ci.iloc[:, 1],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=ci.index, y=ci.iloc[:, 0],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=colors[i % len(colors)].replace('#', '#33') + '33',
                showlegend=False,
                hoverinfo='skip'
            ))

            median = kmf.median_survival_time_
            summary_lines.append(f"**{g}**: Median = {median:.1f}{' (censored)' if np.isnan(median) else ''}")

        if len(groups) == 2:
            g1 = df[df[group_col] == groups[0]]
            g2 = df[df[group_col] == groups[1]]
            lr = logrank_test(g1[time_col], g2[time_col], g1[event_col], g2[event_col])
            p_value = lr.p_value
            summary_lines.append(f"**Log-rank p-value**: {p_value:.4f}")

    else:
        kmf.fit(df[time_col], df[event_col], label="Overall")
        sf = kmf.survival_function_
        ci = kmf.confidence_interval_

        fig.add_trace(go.Scatter(
            x=sf.index, y=sf.iloc[:, 0],
            mode='lines',
            name="Overall",
            line=dict(color="#1f77b4", width=3),
            hovertemplate="Time: %{x}<br>Survival: %{y:.3f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=ci.index, y=ci.iloc[:, 1], line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=ci.index, y=ci.iloc[:, 0], line=dict(width=0), fill='tonexty',
            fillcolor="#1f77b433", showlegend=False, hoverinfo='skip'
        ))
        median = kmf.median_survival_time_
        summary_lines.append(f"**Overall Median Survival**: {median:.1f}")

    fig.update_layout(
        title="Kaplan-Meier Survival Curve",
        xaxis_title="Time",
        yaxis_title="Survival Probability",
        yaxis=dict(range=[0, 1.05]),
        hovermode="x unified",
        template="simple_white",
        legend=dict(title=group_col or "Group")
    )

    summary = "<br>".join(summary_lines)
    if p_value and p_value < 0.05:
        summary += " <span style='color:red'>(Significant difference)</span>"

    return fig, summary

# --- Plot: Forest Plot (Cox HR) (Unchanged) ---
def plot_forest_plot(cox_results: pd.DataFrame) -> go.Figure:
    """Generate forest plot from Cox model summary."""
    df = cox_results.copy()
    df = df[df['covariate'] != 'baseline']
    df['HR'] = np.exp(df['coef'])
    df['lower'] = np.exp(df['coef'] - 1.96 * df['std_err'])
    df['upper'] = np.exp(df['coef'] + 1.96 * df['std_err'])
    df['p'] = df['p']
    df = df.sort_values('HR')

    fig = go.Figure()

    for i, row in df.iterrows():
        color = 'red' if row['p'] < 0.05 else 'black'
        fig.add_trace(go.Scatter(
            x=[row['lower'], row['upper']],
            y=[i, i],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False,
            hovertemplate=f"<b>{row['covariate']}</b><br>HR: {row['HR']:.2f}<br>95% CI: [{row['lower']:.2f}, {row['upper']:.2f}]<br>p = {row['p']:.4f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=[row['HR']], y=[i],
            mode='markers',
            marker=dict(color=color, size=10, symbol='square'),
            showlegend=False
        ))

    fig.add_vline(x=1, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Forest Plot of Hazard Ratios (Cox Model)",
        xaxis_title="Hazard Ratio (95% CI)",
        yaxis=dict(
            tickvals=list(range(len(df))),
            ticktext=[f"{c} (HR={hr:.2f})" for c, hr in zip(df['covariate'], df['HR'])],
            autorange="reversed"
        ),
        xaxis=dict(type="log", range=[np.log(0.1), np.log(10)]),
        height=300 + len(df) * 50,
        template="simple_white"
    )
    return fig

# --- Fit Cox Model (Unchanged) ---
def fit_cox_model(df: pd.DataFrame, time_col: str, event_col: str, covariates: List[str]) -> Dict:
    """Fit Cox model and return summary + forest data."""
    required = [time_col, event_col] + covariates
    if not all(c in df.columns for c in required):
        return {"error": "Missing required columns for Cox model."}

    # Drop rows with missing values in model vars
    model_df = df[required].dropna()

    if len(model_df) < 10:
        return {"error": "Not enough data for Cox model."}

    cph = CoxPHFitter()
    cph.fit(model_df, duration_col=time_col, event_col=event_col)

    summary = cph.summary
    summary['covariate'] = summary.index
    summary = summary.reset_index(drop=True)

    # Add baseline if categorical
    baseline = None
    if any(df[c].dtype == 'object' for c in covariates):
        model_df_encoded = pd.get_dummies(model_df, drop_first=True)
        cph.fit(model_df_encoded, duration_col=time_col, event_col=event_col)
        summary = cph.summary
        summary['covariate'] = summary.index
        summary = summary.reset_index(drop=True)

    # Add concordance, AIC
    stats = {
        "concordance": cph.concordance_index_,
        "AIC": cph.AIC_,
        "n": len(model_df),
        "events": model_df[event_col].sum()
    }

    return {
        "summary_df": summary,
        "stats": stats,
        "model_df": model_df
    }

# --- Agent Prompt (Unchanged) ---
prompt = PromptTemplate.from_template("""
You are a survival analysis expert using Databricks.

Task:
1. Parse user request for:
   - Table
   - Time column
   - Event column (0/1)
   - Group column (optional)
   - Covariates for Cox (optional, e.g., age, sex, stage)
2. Generate **one** natural language query to fetch **only** required columns.
3. Call `query_databricks_genie`.
4. From result:
   - Convert `data` to DataFrame
   - Validate: time > 0, event in [0,1], no NaN in key cols
5. Generate:
   - Kaplan-Meier plot (with CI)
   - If group: log-rank test
   - If covariates: Cox model + forest plot
6. Return **JSON only**:
   {{
     "km_plot": true,
     "cox_plot": true/false,
     "summary": "text",
     "km_figure": {{...plotly json...}},
     "cox_figure": {{...plotly json...}},
     "stats": {{...}}
   }}

Tools: {tool_names}
{tools}

User: {input}
""")

# --- Agent Setup (Unchanged) ---
tools = [query_databricks_genie]
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

# --- Streamlit App (Unchanged) ---
st.set_page_config(page_title="Survival Agent", layout="wide")
st.title("Survival Analysis Agent")
st.caption("ChatDatabricks + Genie (via WorkspaceClient) + lifelines + Plotly")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "km_plot" in msg:
            st.plotly_chart(msg["km_plot"], use_container_width=True)
        if "cox_plot" in msg:
            st.plotly_chart(msg["cox_plot"], use_container_width=True)
        if "stats" in msg:
            st.caption(msg["stats"])

# Input
if user_input := st.chat_input("e.g., KM curve for 'lung_cancer' table, time='months', event='death', group='treatment', cox with 'age', 'stage'"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Running analysis with DBRX..."):
            try:
                result = agent_executor.invoke({"input": user_input})
                output = result["output"]

                try:
                    parsed = json.loads(output) if isinstance(output, str) else output
                except:
                    parsed = {"summary": output}

                summary = parsed.get("summary", "Analysis complete.")
                st.markdown(summary)

                # KM Plot
                if "km_figure" in parsed:
                    km_fig = go.Figure(parsed["km_figure"])
                    st.plotly_chart(km_fig, use_container_width=True)
                    # Note: For full integration, call plot_kaplan_meier_plotly here if needed
                    st.caption("Kaplan-Meier summary rendered.")

                # Cox + Forest
                if "cox_figure" in parsed:
                    cox_fig = go.Figure(parsed["cox_figure"])
                    st.plotly_chart(cox_fig, use_container_width=True)

                # Stats
                stats = parsed.get("stats", "")
                if stats:
                    st.json(stats, expanded=False)

                # Save
                msg = {"role": "assistant", "content": summary}
                if "km_figure" in parsed:
                    msg["km_plot"] = km_fig
                if "cox_figure" in parsed:
                    msg["cox_plot"] = cox_fig
                if stats:
                    msg["stats"] = stats
                st.session_state.messages.append(msg)

            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})