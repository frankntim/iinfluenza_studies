import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import uuid
import asyncio

from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import Tool
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler

# Load CSV
df = pd.read_csv("titanic.csv")

# Streaming text and plot state
streamed_content = {"text": ""}
streamed_plot = {"fig": None}

# Define a plotting tool
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

plot_tool = Tool(
    name="ChartGenerator",
    func=plot_chart,
    description="Generates plots like 'age distribution', 'survival by class', or 'fare vs age'."
)

# Setup LangChain agent
llm = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)
agent = create_pandas_dataframe_agent(llm, df, extra_tools=[plot_tool], verbose=True)

# Callback for streaming tokens
class StreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        streamed_content["text"] += token

# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Chatbot with Streaming + Plot"

app.layout = html.Div([
    dbc.Button("Open Chatbot", id="open", n_clicks=0),
    dcc.Store(id="stream-update", data="", storage_type="memory"),

    dbc.Modal([
        dbc.ModalHeader("Titanic Chatbot"),
        dbc.ModalBody([
            html.Div(id="chat-output", style={"whiteSpace": "pre-wrap", "minHeight": "200px"}),
            dcc.Input(id="user-input", type="text", placeholder="Ask a question...", className="form-control"),
            dbc.Button("Send", id="send", n_clicks=0, color="primary", className="mt-2"),
            dcc.Graph(id="chat-graph", style={"marginTop": "20px"})
        ]),
        dbc.ModalFooter(dbc.Button("Close", id="close", className="ms-auto", n_clicks=0)),
    ], id="modal", is_open=False),
])

# Modal toggling
@app.callback(
    Output("modal", "is_open", allow_duplicate=True),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
    prevent_initial_call="initial_duplicate"
)
def toggle_modal(open_clicks, close_clicks, is_open):
    return not is_open if ctx.triggered_id in ["open", "close"] else is_open

# Trigger agent and stream response (async)
@app.callback(
    Output("stream-update", "data", allow_duplicate=True),
    Input("send", "n_clicks"),
    State("user-input", "value"),
    prevent_initial_call="initial_duplicate"
)
def trigger_agent(n, user_query):
    if not user_query:
        return ""
    
    streamed_content["text"] = ""
    streamed_plot["fig"] = None

    async def run_agent():
        await agent.ainvoke(
            [HumanMessage(content=user_query)],
            config={"callbacks": [StreamingHandler()]}
        )

    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.ensure_future(run_agent())
    else:
        loop.run_until_complete(run_agent())

    return str(uuid.uuid4())

# Display streaming text output
@app.callback(
    Output("chat-output", "children"),
    Input("stream-update", "data")
)
def update_output(_):
    return streamed_content["text"]

# Display plot (if any)
@app.callback(
    Output("chat-graph", "figure"),
    Output("chat-graph", "style"),
    Input("stream-update", "data")
)
def update_plot(_):
    if streamed_plot["fig"]:
        return streamed_plot["fig"], {"display": "block"}
    return {}, {"display": "none"}

if __name__ == "__main__":
    app.run_server(debug=True)
