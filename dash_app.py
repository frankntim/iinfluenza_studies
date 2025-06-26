import dash
from dash import html, dcc, Input, Output, State, ctx
import pandas as pd
import uuid
import os
from langchain.agents.agent_types import AgentType
from langchain.experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler, BaseCallbackHandler

# Load Titanic CSV
df = pd.read_csv('titanic.csv')

# Use in-memory storage for chat streaming
STREAM_CACHE = {}

# Custom streaming handler for Dash UI
class DashStreamHandler(BaseCallbackHandler):
    def __init__(self, session_id):
        self.session_id = session_id
        STREAM_CACHE[self.session_id] = ""

    def on_llm_new_token(self, token: str, **kwargs):
        STREAM_CACHE[self.session_id] += token

# LangChain Agent Setup
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    streaming=True,
)

# Dash App Setup
app = dash.Dash(__name__)
app.title = "Titanic Chatbot Modal"
app.layout = html.Div([
    html.Button("Open Chatbot", id="open-modal", n_clicks=0),
    dcc.Store(id="session-id", data=str(uuid.uuid4())),
    dcc.Interval(id="stream-update", interval=500, n_intervals=0, disabled=True),
    html.Div(id="stream-output", style={"whiteSpace": "pre-wrap", "marginTop": "1rem"}),

    html.Div([
        html.Div([
            html.H2("Titanic Chatbot", style={"marginBottom": "10px"}),
            dcc.Textarea(id="user-input", style={"width": "100%", "height": "100px"}, placeholder="Ask me about the Titanic dataset..."),
            html.Button("Send", id="send-btn", n_clicks=0),
            html.Div(id="chat-log", style={"whiteSpace": "pre-wrap", "height": "300px", "overflowY": "auto", "marginTop": "1rem"}),
            html.Button("Close", id="close-modal", n_clicks=0, style={"marginTop": "10px"}),
        ], style={
            "backgroundColor": "white",
            "padding": "20px",
            "borderRadius": "10px",
            "width": "600px",
            "maxWidth": "90%",
            "boxShadow": "0 0 10px rgba(0, 0, 0, 0.2)"
        }),
    ], id="modal", style={
        "display": "none",
        "position": "fixed",
        "top": 0, "left": 0, "right": 0, "bottom": 0,
        "backgroundColor": "rgba(0, 0, 0, 0.5)",
        "justifyContent": "center",
        "alignItems": "center",
        "zIndex": 1000,
    })
])

# Modal open/close
@app.callback(
    Output("modal", "style"),
    Input("open-modal", "n_clicks"),
    Input("close-modal", "n_clicks"),
    prevent_initial_call=True
)
def toggle_modal(open_clicks, close_clicks):
    if ctx.triggered_id == "open-modal":
        return {"display": "flex", "position": "fixed", "top": 0, "left": 0, "right": 0, "bottom": 0,
                "backgroundColor": "rgba(0, 0, 0, 0.5)", "justifyContent": "center", "alignItems": "center", "zIndex": 1000}
    return {"display": "none"}

# Trigger LLM streaming
@app.callback(
    Output("stream-update", "disabled"),
    Input("send-btn", "n_clicks"),
    State("user-input", "value"),
    State("session-id", "data"),
    prevent_initial_call=True
)
def send_query(n, user_input, session_id):
    if not user_input:
        return True

    # Clear stream
    STREAM_CACHE[session_id] = ""

    # Launch async agent invoke
    def run_agent():
        handler = DashStreamHandler(session_id=session_id)
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            handle_parsing_errors=True
        )
        try:
            agent.invoke({"input": user_input}, config={"callbacks": [handler]})
        except Exception as e:
            STREAM_CACHE[session_id] += f"\n[ERROR]: {str(e)}"

    import threading
    threading.Thread(target=run_agent).start()

    return False  # Enable Interval for streaming

# Stream output updates
@app.callback(
    Output("chat-log", "children"),
    Output("stream-output", "children"),
    Output("stream-update", "disabled"),
    Input("stream-update", "n_intervals"),
    State("session-id", "data"),
    State("user-input", "value"),
    prevent_initial_call=True
)
def update_stream(n, session_id, user_input):
    text = STREAM_CACHE.get(session_id, "")
    # Stop interval if stream is done
    if text.endswith(('.', '\n')) and len(text) > 5:
        return text, "", True
    return text, text, False

if __name__ == '__main__':
    app.run_server(debug=True)
