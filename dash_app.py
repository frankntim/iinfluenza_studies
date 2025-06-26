import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import uuid
import asyncio
import threading

from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import Tool
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler

# === Load Titanic CSV ===
df = pd.read_csv("titanic.csv")

# === Shared global state ===
streamed_tokens = []  # live token buffer
streamed_plot = {"fig": None}
is_streaming = {"active": False}

# === Define plot generation tool ===
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

# === LangChain Agent ===
llm = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)
agent = create_pandas_dataframe_agent(llm, df, extra_tools=[plot_tool], verbose=True)

# === Streaming callback ===
class StreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        streamed_tokens.append(token)

# === Dash app ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Streaming Chatbot with Plotting"

app.layout = html.Div([
    dbc.Button("Open Chatbot", id="open", n_clicks=0),
    dcc.Store(id="chat-text", data=""),
    dcc.Interval(id="poll-stream", interval=250, n_intervals=0),

    dbc.Modal([
        dbc.ModalHeader("Titanic CSV Chatbot"),
        dbc.ModalBody([
            html.Div(id="chat-output", style={"whiteSpace": "pre-wrap", "minHeight": "200px"}),
            dcc.Input(id="user-input", type="text", placeholder="Ask a question...", className="form-control"),
            dbc.Button("Send", id="send", n_clicks=0, color="primary", className="mt-2"),
            dcc.Graph(id="chat-graph", style={"marginTop": "20px"})
        ]),
        dbc.ModalFooter(dbc.Button("Close", id="close", className="ms-auto", n_clicks=0)),
    ], id="modal", is_open=False),
])

# === Modal toggle ===
@app.callback(
    Output("modal", "is_open", allow_duplicate=True),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
    prevent_initial_call="initial_duplicate"
)
def toggle_modal(open_clicks, close_clicks, is_open):
    return not is_open if ctx.triggered_id in ["open", "close"] else is_open

# === Start streaming response ===
@app.callback(
    Output("chat-text", "data"),
    Input("send", "n_clicks"),
    State("user-input", "value"),
    prevent_initial_call=True
)
def trigger_agent(n, user_query):
    if not user_query:
        return ""
    streamed_tokens.clear()
    streamed_plot["fig"] = None
    is_streaming["active"] = True

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(agent.ainvoke(
            [HumanMessage(content=user_query)],
            config={"callbacks": [StreamingHandler()]}
        ))
        is_streaming["active"] = False

    threading.Thread(target=run).start()
    return ""  # Start from blank

# === Update chat output every 250ms ===
@app.callback(
    Output("chat-output", "children"),
    Output("chat-graph", "figure"),
    Output("chat-graph", "style"),
    Input("poll-stream", "n_intervals"),
    State("chat-text", "data"),
    prevent_initial_call=True
)
def stream_to_output(_, current_text):
    if streamed_tokens:
        current_text += "".join(streamed_tokens)
        streamed_tokens.clear()
    fig = streamed_plot["fig"]
    return current_text, fig if fig else {}, {"display": "block" if fig else "none"}

# === Run app ===
if __name__ == "__main__":
    app.run_server(debug=True)
