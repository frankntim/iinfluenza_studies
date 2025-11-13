import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.genai import GenieAPI
from langchain_databricks import ChatDatabricks
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import json

# --- Configuration ---
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID")
MODEL_ENDPOINT = os.getenv("DATABRICKS_MODEL_ENDPOINT", "databricks-dbrx-instruct")

if not all([DATABRICKS_HOST, DATABRICKS_TOKEN, GENIE_SPACE_ID]):
    st.error("Please set DATABRICKS_HOST, DATABRICKS_TOKEN, and GENIE_SPACE_ID environment variables.")
    st.stop()

# --- Initialize Databricks Clients ---
client = WorkspaceClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)
genie = GenieAPI(client)

# --- LLM: ChatDatabricks ---
llm = ChatDatabricks(
    endpoint=MODEL_ENDPOINT,
    temperature=0.1,
    max_tokens=1024
)

# --- Custom Tool: Query Genie ---
@tool
def query_databricks_genie(natural_language_query: str) -> Dict[str, Any]:
    """
    Query Databricks Genie with natural language and return:
    - 'data': List of dicts (JSON-serializable)
    - 'sql': Generated SQL
    - 'error': If failed
    """
    try:
        with st.spinner(f"Querying Genie: {natural_language_query}"):
            message = genie.start_conversation_and_wait(
                space_id=GENIE_SPACE_ID,
                content=natural_language_query
            )

        if not message.attachments:
            return {"error": "No results returned from Genie."}

        attachment = message.attachments[0]
        result = genie.get_message_attachment_query_result(
            space_id=GENIE_SPACE_ID,
            conversation_id=message.conversation_id,
            message_id=message.message_id,
            attachment_id=attachment.attachment_id
        )

        # Convert to DataFrame
        if result.rows and result.columns:
            df = pd.DataFrame(result.rows, columns=result.columns)
        else:
            df = pd.DataFrame()

        return {
            "data": df.to_dict(orient="records"),
            "sql": attachment.sql,
            "row_count": len(df)
        }

    except Exception as e:
        return {"error": str(e)}

# --- Survival Plotting Function ---
def plot_kaplan_meier(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str = None
) -> Dict[str, Any]:
    """
    Fit Kaplan-Meier and return Plotly figure + stats.
    """
    kmf = KaplanMeierFitter()
    fig = go.Figure()

    if group_col and group_col in df.columns:
        groups = df[group_col].dropna().unique()
        results = []

        for group in groups:
            mask = df[group_col] == group
            group_df = df[mask]
            if len(group_df) == 0:
                continue

            kmf.fit(
                durations=group_df[time_col],
                event_observed=group_df[event_col],
                label=str(group)
            )
            kmf.plot_survival_function(ax=None, ci_show=False)
            fig.add_trace(go.Scatter(
                x=kmf.timeline,
                y=kmf.survival_function_.iloc[:, 0],
                mode='lines',
                name=str(group),
                hovertemplate=
                f"<b>{group}</b><br>" +
                "Time: %{x}<br>" +
                "Survival: %{y:.3f}<extra></extra>"
            ))
            results.append((group, kmf.median_survival_time_))

        # Log-rank test if >1 group
        if len(groups) > 1:
            group1 = df[df[group_col] == groups[0]]
            group2 = df[df[group_col] == groups[1]]
            logrank = logrank_test(
                group1[time_col], group2[time_col],
                group1[event_col], group2[event_col]
            )
            p_value = logrank.p_value
        else:
            p_value = None

        median_text = "<br>".join([f"{g}: {m:.1f}" for g, m in results if pd.notna(m)])
        p_text = f"<br>Log-rank p-value: {p_value:.4f}" if p_value else ""

    else:
        kmf.fit(df[time_col], event_observed=df[event_col], label="Overall")
        fig.add_trace(go.Scatter(
            x=kmf.timeline,
            y=kmf.survival_function_.iloc[:, 0],
            mode='lines',
            name="Overall",
            line=dict(width=3)
        ))
        median_text = f"Median survival: {kmf.median_survival_time_:.1f}"
        p_text = ""

    fig.update_layout(
        title="Kaplan-Meier Survival Curve",
        xaxis_title="Time",
        yaxis_title="Survival Probability",
        yaxis=dict(range=[0, 1.05]),
        hovermode="x unified",
        template="simple_white",
        legend=dict(title=group_col or "Group")
    )

    summary = f"**Median Survival Time(s):**<br>{median_text}{p_text}"
    return {"figure": fig, "summary": summary}

# --- Agent Prompt ---
prompt = PromptTemplate.from_template("""
You are a survival analysis expert using Databricks Genie.

Your task:
1. Understand the user’s request for survival analysis.
2. Identify:
   - Table or dataset
   - Time column (e.g., days_to_event)
   - Event column (0 = censored, 1 = event)
   - Optional: Grouping column (e.g., treatment)
3. Generate a **natural language query** for Genie to fetch ONLY the required columns.
4. Call `query_databricks_genie` with that query.
5. From the result, extract `data` and convert to DataFrame.
6. Validate: time > 0, event in [0,1], no missing critical values.
7. Call `plot_kaplan_meier` with correct column names.
8. Return:
   - Natural language summary
   - Plot
   - Key stats (median survival, p-value if grouped)

**Only return final answer in JSON format** with keys: `summary`, `plot`, `stats`.

Tools: {tool_names}
{tools}

User request: {input}
""")

# --- Create Agent ---
tools = [query_databricks_genie]
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

# --- Streamlit UI ---
st.set_page_config(page_title="Survival Curve Agent", layout="wide")
st.title("Survival Curve Analysis Agent")
st.caption("Powered by ChatDatabricks + Genie + lifelines")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "plot" in msg:
            st.plotly_chart(msg["plot"], use_container_width=True)
        if "stats" in msg:
            st.caption(msg["stats"])

# User input
if user_input := st.chat_input("e.g., Plot survival curve for lung cancer patients in table 'trials' using 'months' and 'death' grouped by 'chemo'"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing with DBRX..."):
            try:
                result = agent_executor.invoke({"input": user_input})
                output = result["output"]

                # Parse JSON output
                try:
                    parsed = json.loads(output) if isinstance(output, str) else output
                except:
                    parsed = {"summary": output}

                summary = parsed.get("summary", "Analysis complete.")
                st.markdown(summary)

                # Plot
                if "plot" in parsed:
                    fig = parsed["plot"]
                    st.plotly_chart(fig, use_container_width=True)

                # Stats
                stats = parsed.get("stats", "")
                if stats:
                    st.caption(stats)

                # Save to history
                msg = {"role": "assistant", "content": summary}
                if "plot" in parsed:
                    msg["plot"] = fig
                if stats:
                    msg["stats"] = stats
                st.session_state.messages.append(msg)

            except Exception as e:
                st.error(f"Agent error: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})