import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid
import asyncio
import threading
import traceback
import plotly.io as pio
from lifelines import KaplanMeierFitter

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

# === Shared state ===
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

# === KM plot with LLM parser ===
class KMArgs(BaseModel):
    time_column: str = Field(description="Time-to-event column")
    event_column: str = Field(description="Event occurred (1) or censored (0)")
    group_column: str = Field(default=None, description="Optional grouping column")

parser = PydanticOutputParser(pydantic_object=KMArgs)

prompt = PromptTemplate(
    template="""
You extract survival analysis parameters from a user query.
Return JSON with: time_column, event_column, group_column (optional).

Query: {query}
{format_instructions}
""",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm_chain = LLMChain(llm=ChatOpenAI(model="gpt-4", temperature=0), prompt=prompt)

def km_survival_plot(query: str):
    try:
        parsed = parser.parse(llm_chain.run(query))
        T_col = parsed.time_column
        E_col = parsed.event_column
        group_col = parsed.group_column

        if T_col not in df.columns or E_col not in df.columns:
            return f"⚠️ Columns `{T_col}` or `{E_col}` not found."

        T = df[T_col]
        E = df[E_col]
        fig = go.Figure()

        if group_col and group_col in df.columns:
            for group, subdf in df.groupby(group_col):
                kmf = KaplanMeierFitter()
                kmf.fit(subdf[T_col], subdf[E_col], label=str(group))
                fig.add_trace(go.Scatter(
                    x=kmf.survival_function_.index,
                    y=kmf.survival_function_[kmf._label],
                    mode="lines", name=str(group)
                ))
            fig.update_layout(title=f"KM Curve by {group_col}", xaxis_title=T_col, yaxis_title="Survival Probability")
        else:
            kmf = KaplanMeierFitter()
            kmf.fit(T, E, label="All")
            fig.add_trace(go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_["All"],
                mode="lines", name="All"
            ))
            fig.update_layout(title="Kaplan-Meier Curve", xaxis_title=T_col, yaxis_title="Survival Probability")

        streamed_plot["fig"] = fig
        return f"Kaplan-Meier plot using `{T_col}` and `{E_col}`" + (f" grouped by `{group_col}`." if group_col else ".")
    except Exception as e:
        return f"❌ Failed to parse or plot: {e}"

# === LangChain tools ===
plot_tool = Tool(
    name="ChartGenerator",
    func=plot_chart,
    description="Generates plots like age distribution, fare vs age, or survival by class."
)

km_tool = Tool(
    name="KaplanMeierPlot",
    func=km_survival_plot,
    description="Generate Kaplan-Meier survival plots. Mention time and event columns, and optionally group by a column."
)

# === LangChain Agent ===
llm = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)
agent = create_pandas_dataframe_agent(llm, df, extra_tools=[plot_tool, km_tool], verbose=True)

# === Streaming callback ===
class StreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        streamed_tokens.append(token)

# === Dash app ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Streaming Chatbot with KM Plots"

app.layout = html.Div([
    dbc.Button("Open Chatbot", id="open", n_clicks=0),
    dcc.Store(id="chat-text", data=[]),
    dcc.Interval(id="poll-stream", interval=250, n_intervals=0),

    dbc.Modal([
        dbc.ModalHeader("Titanic CSV Chatbot"),
        dbc.ModalBody([
            html.Div(id="chat-output", style={
                "whiteSpace": "pre-wrap",
                "overflowY": "scroll",
                "maxHeight": "300px",
                "border": "1px solid #ccc",
                "padding": "10px",
                "marginBottom": "10px"
            }),
            dcc.Input(id="user-input", type="text", placeholder="Ask a question...", className="form-control"),
            dbc.Button("Send", id="send", n_clicks=0, color="primary", className="mt-2"),
            dcc.Graph(id="chat-graph", style={"marginTop": "20px"})
        ]),
        dbc.ModalFooter(dbc.Button("Close", id="close", className="ms-auto", n_clicks=0)),
    ], id="modal", is_open=False),
])

# === Toggle modal ===
@app.callback(
    Output("modal", "is_open", allow_duplicate=True),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
    prevent_initial_call="initial_duplicate"
)
def toggle_modal(open_clicks, close_clicks, is_open):
    return not is_open if ctx.triggered_id in ["open", "close"] else is_open

# === Agent trigger ===
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

# === Stream updates ===
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

# === Run app ===
if __name__ == "__main__":
    app.run_server(debug=True)
