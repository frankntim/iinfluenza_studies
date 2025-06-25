import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
import threading

# Load Titanic CSV
df = pd.read_csv("titanic.csv")

# Streaming handler
class StreamingHandler(BaseCallbackHandler):
    def __init__(self):
        self.chunks = []
        self.lock = threading.Lock()
        self.done = False

    def on_llm_new_token(self, token: str, **kwargs):
        with self.lock:
            self.chunks.append(token)

    def on_llm_end(self, response, **kwargs):
        with self.lock:
            self.done = True

    def get_text(self):
        with self.lock:
            return "".join(self.chunks)

    def is_done(self):
        with self.lock:
            return self.done

    def reset(self):
        with self.lock:
            self.chunks.clear()
            self.done = False

# Initialize agent
stream_handler = StreamingHandler()
llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True, callbacks=[stream_handler])
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=False,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

# App setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    dbc.Button("Open Titanic Chat", id="open-chat", n_clicks=0),

    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Titanic Data Chat")),
        dbc.ModalBody([
            html.Div(id='chat-log', style={
                'height': '300px', 'overflowY': 'auto',
                'border': '1px solid #ccc', 'padding': '10px',
                'marginBottom': '10px', 'backgroundColor': '#f9f9f9'
            }),
            dcc.Textarea(
                id='user-input',
                placeholder='Ask a question about Titanic data...',
                style={'width': '100%', 'height': 100}
            ),
        ]),
        dbc.ModalFooter([
            dbc.Button("Send", id="send-button", n_clicks=0),
            dbc.Button("Close", id="close-chat", className="ms-auto", n_clicks=0),
        ]),
    ], id="chat-modal", is_open=False, size="lg"),

    dcc.Interval(id="stream-interval", interval=500, n_intervals=0, disabled=True),
    dcc.Store(id="chat-history", data=[]),
    dcc.Store(id="pending", data=False),
    dcc.Store(id="stream-buffer", data=""),
])

# Toggle modal
@app.callback(
    Output("chat-modal", "is_open"),
    [Input("open-chat", "n_clicks"), Input("close-chat", "n_clicks")],
    [State("chat-modal", "is_open")]
)
def toggle_modal(open_click, close_click, is_open):
    if open_click or close_click:
        return not is_open
    return is_open

# Start agent on send
@app.callback(
    Output("pending", "data"),
    Output("chat-history", "data"),
    Output("user-input", "value"),
    Output("stream-buffer", "data"),
    Input("send-button", "n_clicks"),
    State("user-input", "value"),
    State("chat-history", "data"),
    prevent_initial_call=True
)
def start_response(n, query, history):
    if not query:
        return False, history, "", ""

    history.append({"sender": "User", "text": query})
    stream_handler.reset()

    def run_query():
        try:
            agent.invoke(query)
        except Exception as e:
            stream_handler.chunks.append(f"\n[Error: {str(e)}]")
            stream_handler.done = True

    threading.Thread(target=run_query).start()
    return True, history, "", ""

# Update UI with stream buffer
@app.callback(
    Output("chat-log", "children"),
    Output("stream-interval", "disabled"),
    Output("stream-buffer", "data"),
    Input("stream-interval", "n_intervals"),
    State("pending", "data"),
    State("chat-history", "data"),
    State("stream-buffer", "data"),
)
def update_stream(n, pending, history, buffer):
    if not pending:
        return format_chat_log(history), True, ""

    current_text = stream_handler.get_text()
    temp_history = history + [{"sender": "Bot", "text": current_text}]

    if stream_handler.is_done():
        return format_chat_log(temp_history), True, current_text

    return format_chat_log(temp_history), False, current_text

# When stream ends, finalize chat history
@app.callback(
    Output("chat-history", "data"),
    Output("pending", "data"),
    Input("stream-buffer", "data"),
    State("chat-history", "data"),
    prevent_initial_call=True
)
def finalize_history(buffer, history):
    if buffer:
        history.append({"sender": "Bot", "text": buffer})
    return history, False

# Format chat history
def format_chat_log(history):
    return html.Div([
        html.Div([
            html.Strong(f"{msg['sender']}: "),
            html.Span(msg['text'])
        ]) for msg in history
    ])

if __name__ == "__main__":
    app.run_server(debug=True)
