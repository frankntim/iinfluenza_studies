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

streamed_tokens = []
streamed_plot = {"fig": None}
streamed_table = {"html": ""}
is_streaming = {"active": False}

# === Chart tool ===
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
    streamed_table["html"] = ""
    return "Here's the chart you requested."

# === KM tool ===
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
        T_col, E_col, group_col = parsed.time_column, parsed.event_column, parsed.group_column
        if T_col not in df.columns or E_col not in df.columns:
            return f"⚠️ `{T_col}` or `{E_col}` not found."
        fig = go.Figure()
        if group_col and group_col in df.columns:
            for g, sub in df.groupby(group_col):
                kmf = KaplanMeierFitter()
                kmf.fit(sub[T_col], sub[E_col], label=str(g))
                fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_[kmf._label], mode="lines", name=str(g)))
            fig.update_layout(title=f"KM Curve by {group_col}", xaxis_title=T_col, yaxis_title="Survival Prob")
        else:
            kmf = KaplanMeierFitter()
            kmf.fit(df[T_col], df[E_col], label="All")
            fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_["All"], mode="lines", name="All"))
            fig.update_layout(title="Kaplan-Meier Curve", xaxis_title=T_col, yaxis_title="Survival Prob")
        streamed_plot["fig"] = fig
        streamed_table["html"] = ""
        return "Kaplan-Meier plot generated."
    except Exception as e:
        return f"❌ KM Error: {e}"

# === Cox tool ===
class CoxArgs(BaseModel):
    time_column: str
    event_column: str
    covariates: list[str]

parser_cox = PydanticOutputParser(pydantic_object=CoxArgs)
prompt_cox = PromptTemplate(
    template="""Extract Cox regression inputs from this query. Return JSON: time_column, event_column, covariates.

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
        T_col, E_col, covs = args.time_column, args.event_column, args.covariates
        for col in [T_col, E_col] + covs:
            if col not in df.columns:
                return f"⚠️ `{col}` not found."
        dfc = df[[T_col, E_col] + covs].dropna()
        dfc = pd.get_dummies(dfc, drop_first=True)
        cph = CoxPHFitter()
        cph.fit(dfc, duration_col=T_col, event_col=E_col)
        summary = cph.summary.reset_index().rename(columns={
            'covariate': 'Variable', 'exp(coef)': 'Hazard Ratio', 'se(coef)': 'Std. Error',
            'p': 'p-value', 'coef lower 95%': 'Lower 95%', 'coef upper 95%': 'Upper 95%'
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=summary['Hazard Ratio'], y=summary['Variable'], orientation='h',
            error_x=dict(type='data', array=summary['Upper 95%'] - summary['Hazard Ratio'],
                         arrayminus=summary['Hazard Ratio'] - summary['Lower 95%'])
        ))
        fig.update_layout(title="Hazard Ratios with 95% CI", xaxis_title="Hazard Ratio", yaxis_title="Variables")
        streamed_plot["fig"] = fig
        streamed_table["html"] = summary.to_html(index=False, classes="table table-striped")
        return "Cox model fitted. Table and plot rendered below."
    except Exception as e:
        return f"❌ CoxPH Error: {e}"

# === Tools ===
tools = [
    Tool(name="ChartGenerator", func=plot_chart, description="Basic charts like age distribution, fare vs age, etc."),
    Tool(name="KaplanMeierPlot", func=km_survival_plot, description="Kaplan-Meier plots with time/event/group."),
    Tool(name="CoxPHFitter", func=coxph_plot, description="Cox regression with time/event/covariates.")
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
            html.Div(id="chat-output", style={"whiteSpace": "pre-wrap", "overflowY": "scroll", "maxHeight": "300px",
                                              "border": "1px solid #ccc", "padding": "10px", "marginBottom": "10px"}),
            dcc.Input(id="user-input", type="text", placeholder="Ask something...", className="form-control"),
            dbc.Button("Send", id="send", n_clicks=0, color="primary", className="mt-2"),
            dcc.Graph(id="chat-graph", style={"marginTop": "20px"}),
            html.Div(id="cox-summary", dangerously_allow_html=True, className="mt-3")
        ]),
        dbc.ModalFooter(dbc.Button("Close", id="close", className="ms-auto", n_clicks=0)),
    ], id="modal", is_open=False),
])

@app.callback(Output("modal", "is_open", allow_duplicate=True),
              [Input("open", "n_clicks"), Input("close", "n_clicks")],
              [State("modal", "is_open")], prevent_initial_call="initial_duplicate")
def toggle_modal(open_clicks, close_clicks, is_open):
    return not is_open if ctx.triggered_id in ["open", "close"] else is_open

@app.callback(Output("chat-text", "data"),
              Input("send", "n_clicks"),
              State("user-input", "value"),
              State("chat-text", "data"),
              prevent_initial_call=True)
def trigger_agent(n, user_query, history):
    if not user_query: return history
    history.append(f"\U0001F464: {user_query}")
    streamed_tokens.clear()
    streamed_plot["fig"] = None
    streamed_table["html"] = ""
    is_streaming["active"] = True
    def run():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(agent.ainvoke([HumanMessage(content=user_query)], config={"callbacks": [StreamingHandler()]}))
        except Exception as e:
            streamed_tokens.append(f"\n⚠️ Error:\n{traceback.format_exc()}")
        finally:
            is_streaming["active"] = False
    threading.Thread(target=run).start()
    history.append("\U0001F916: ")
    return history

@app.callback(
    Output("chat-output", "children"),
    Output("chat-graph", "figure"),
    Output("chat-graph", "style"),
    Output("cox-summary", "children"),
    Input("poll-stream", "n_intervals"),
    State("chat-text", "data"),
    prevent_initial_call=True)
def stream_to_output(_, chat_history):
    if chat_history and chat_history[-1].startswith("\U0001F916: "):
        chat_history[-1] = "\U0001F916: " + "".join(streamed_tokens)
    fig = streamed_plot["fig"]
    table_html = streamed_table["html"]
    return ("\n".join(chat_history),
            fig if fig else dash.no_update,
            {"display": "block"} if fig else {"display": "none"},
            html.Div([html.H5("Cox Model Summary"), html.Div(table_html, dangerously_allow_html=True)]) if table_html else "")

if __name__ == "__main__":
    app.run_server(debug=True)
