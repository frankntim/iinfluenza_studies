import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import threading
import traceback
import plotly.io as pio
from lifelines import KaplanMeierFitter, CoxPHFitter

from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import Tool
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

pio.renderers.default = "svg"

# === Load Titanic CSV ===
df = pd.read_csv("titanic.csv")
df['Survived'] = df['Survived'].astype(int)
df['Pclass'] = df['Pclass'].astype(str)

# === Global shared state ===
streamed_tokens = []
streamed_plot = {"fig": None}
is_streaming = {"active": False}

# === General plot tool ===
def plot_chart(query: str):
    query = query.lower()
    if "age distribution" in query:
        fig = px.histogram(df, x="Age", nbins=30, title="Age Distribution")
    elif "survival by class" in query:
        fig = px.histogram(df, x="Pclass", color="Survived", barmode="group", title="Survival by Class")
    elif "fare vs age" in query:
        fig = px.scatter(df, x="Age", y="Fare", color="Survived", title="Fare vs Age")
    else:
        return "Sorry, I couldn't generate a chart for that."
    streamed_plot["fig"] = fig
    return "Here's the chart you requested."

# === KM plot tool using LLM column extraction ===
class KMArgs(BaseModel):
    time_column: str
    event_column: str
    group_column: str = None

parser_km = PydanticOutputParser(pydantic_object=KMArgs)
prompt_km = PromptTemplate(
    template="""Extract Kaplan-Meier arguments from user query as JSON: time_column, event_column, and optional group_column.

Query: {query}
{format_instructions}
""",
    input_variables=["query"],
    partial_variables={"format_instructions": parser_km.get_format_instructions()}
)
llm_chain_km = LLMChain(llm=ChatOpenAI(model="gpt-4", temperature=0), prompt=prompt_km)

def km_survival_plot(query: str):
    try:
        parsed = parser_km.parse(llm_chain_km.run(query))
        T_col = parsed.time_column
        E_col = parsed.event_column
        group_col = parsed.group_column

        if T_col not in df.columns or E_col not in df.columns:
            return f"⚠️ Columns `{T_col}` or `{E_col}` not found."

        fig = go.Figure()
        if group_col and group_col in df.columns:
            for group, subdf in df.groupby(group_col):
                kmf = KaplanMeierFitter()
                kmf.fit(subdf[T_col], event_observed=subdf[E_col], label=str(group))
                fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_[kmf._label], mode="lines", name=str(group)))
            fig.update_layout(title=f"KM Curve by {group_col}", xaxis_title=T_col, yaxis_title="Survival Probability")
        else:
            kmf = KaplanMeierFitter()
            kmf.fit(df[T_col], event_observed=df[E_col], label="All")
            fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_["All"], mode="lines", name="All"))
            fig.update_layout(title="Kaplan-Meier Curve", xaxis_title=T_col, yaxis_title="Survival Probability")

        streamed_plot["fig"] = fig
        return "Kaplan-Meier plot generated."
    except Exception as e:
        return f"❌ KM Error: {e}"

# === Cox PH plot using LLM ===
class CoxArgs(BaseModel):
    time_column: str
    event_column: str
    covariates: list[str]

parser_cox = PydanticOutputParser(pydantic_object=CoxArgs)
prompt_cox = PromptTemplate(
    template="""Extract Cox regression inputs from this query.
Return JSON: time_column, event_column, covariates (a list of strings).

Query: {query}
{format_instructions}
""",
    input_variables=["query"],
    partial_variables={"format_instructions": parser_cox.get_format_instructions()}
)
llm_chain_cox = LLMChain(llm=ChatOpenAI(model="gpt-4", temperature=0), prompt=prompt_cox)

def coxph_plot(query: str):
    try:
        args = parser_cox.parse(llm_chain_cox.run(query))
        T_col = args.time_column
        E_col = args.event_column
        covs = args.covariates

        for col in [T_col, E_col] + covs:
            if col not in df.columns:
                return f"⚠️ Column `{col}` not found."

        df_cox = df[[T_col, E_col] + covs].dropna()
        df_cox = pd.get_dummies(df_cox, drop_first=True)  # handle categorical

        cph = CoxPHFitter()
        cph.fit(df_cox, duration_col=T_col, event_col=E_col)

        fig = go.Figure()
        summary = cph.summary.reset_index()
        for i, row in summary.iterrows():
            fig.add_trace(go.Bar(
                x=[row["coef"]],
                y=[row["index"]],
                orientation='h',
                error_x=dict(type='data', array=[row["se(coef)"]]),
                name=row["index"]
            ))

        fig.update_layout(title="CoxPH Coefficients", xaxis_title="Coefficient", yaxis_title="Covariate")
        streamed_plot["fig"] = fig
        return "Cox model fitted and coefficients plotted."
    except Exception as e:
        return f"❌ CoxPH Error: {e}"

# === LangChain tools ===
tools = [
    Tool(name="ChartGenerator", func=plot_chart, description="Basic charts like age distribution, fare vs age, etc."),
    Tool(name="KaplanMeierPlot", func=km_survival_plot, description="Kaplan-Meier plots with time/event/group."),
    Tool(name="CoxPHFitter", func=coxph_plot, description="Cox regression. Provide time, event, and covariates list.")
]

llm_stream = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)
agent = create_pandas_dataframe_agent(llm_stream, df, extra_tools=tools, verbose=True)

class StreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        streamed_tokens.append(token)

# === Dash App ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Chatbot: KM + CoxPH"

app.layout = html.Div([
    dbc.Button("Open Chatbot", id="open", n_clicks=0),
    dcc.Store(id="chat-text", data=[]),
    dcc.Interval(id="poll-stream", interval=250, n_intervals=0),
    dbc.Modal([
        dbc.ModalHeader("Titanic Chatbot"),
        dbc.ModalBody([
            html.Div(id="chat-output", style={
                "whiteSpace": "pre-wrap",
                "overflowY": "scroll",
                "maxHeight": "300px",
                "border": "1px solid #ccc", "padding": "10px", "marginBottom": "10px"
            }),
            dcc.Input(id="user-input", type="text", placeholder="Ask something...", className="form-control"),
            dbc.Button("Send", id="send", n_clicks=0, color="primary", className="mt-2"),
            dcc.Graph(id="chat-graph", style={"marginTop": "20px"})
        ]),
        dbc.ModalFooter(dbc.Button("Close", id="close", className="ms-auto", n_clicks=0)),
    ], id="modal", is_open=False),
])

@app.callback(
    Output("modal", "is_open", allow_duplicate=True),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
    prevent_initial_call="initial_duplicate"
)
def toggle_modal(open_clicks, close_clicks, is_open):
    return not is_open if ctx.triggered_id in ["open", "close"] else is_open

@app.callback(
    Output("chat-text", "data"),
    Input("send", "n_clicks"),
    State("user-input", "value"),
    State("chat-text", "data"),
    prevent_initial_call=True
)
def trigger_agent(n, user_query, history):
    if not user_query:
        return history
    history.append(f"👤: {user_query}")
    streamed_tokens.clear()
    streamed_plot["fig"] = None
    is_streaming["active"] = True

    def run():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(agent.ainvoke(
                [HumanMessage(content=user_query)],
                config={"callbacks": [StreamingHandler()]}
            ))
        except Exception as e:
            streamed_tokens.append(f"\n⚠️ Error:\n{traceback.format_exc()}")
        finally:
            is_streaming["active"] = False

    threading.Thread(target=run).start()
    history.append("🤖: ")
    return history

@app.callback(
    Output("chat-output", "children"),
    Output("chat-graph", "figure"),
    Output("chat-graph", "style"),
    Input("poll-stream", "n_intervals"),
    State("chat-text", "data"),
    prevent_initial_call=True
)
def stream_to_output(_, chat_history):
    if chat_history and chat_history[-1].startswith("🤖:"):
        chat_history[-1] = "🤖: " + "".join(streamed_tokens)

    fig = streamed_plot["fig"]
    return "\n".join(chat_history), fig if fig else dash.no_update, {"display": "block"} if fig else {"display": "none"}

if __name__ == "__main__":
    app.run_server(debug=True)
