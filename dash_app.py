import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
import threading

# Load Titanic dataset
df = pd.read_csv("titanic.csv")

# Custom handler for streaming
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

# Initialize LangChain agent
stream_handler = StreamingHandler()
llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True, callbacks=[stream_handler])
agent = create_pandas_dataframe_agent(llm=llm, df=df, verbose=False, agent_type=AgentType.OPENAI_FUNCTIONS)

# Initialize Dash app
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

    dcc.Interval(id="stream-interval", interval=300, n_intervals=0, disabled=True),
    dcc.Store(id="chat-history", data=[]),
    dcc.Store(id="streaming", data=False),
    dcc.Store(id="new-query", data=None),
])

# Toggle chat modal
@app.callback(
    Output("chat-modal", "is_open"),
    [Input("open-chat", "n_clicks"), Input("close-chat", "n_clicks")],
    [State("chat-modal", "is_open")]
)
def toggle_modal(open_click, close_click, is_open):
    return not is_open if open_click or close_click else is_open

# Save new query & user message
@app.callback(
    Output("new-query", "data"),
    Output("user-input", "value"),
    Input("send-button", "n_clicks"),
    State("user-input", "value"),
    prevent_initial_call=True
)
def store_query(n, query):
    return query, ""

# Streaming and agent execution logic in one callback
@app.callback(
    Output("chat-log", "children"),
    Output("chat-history", "data"),
    Output("streaming", "data"),
    Output("stream-interval", "disabled"),
    Input("stream-interval", "n_intervals"),
    State("new-query", "data"),
    State("chat-history", "data"),
    State("streaming", "data"),
    prevent_initial_call=True
)
def stream_response(_, new_query, history, streaming):
    # If not streaming, but a query is present → start thread
    if not streaming and new_query:
        # Store user message
        history.append({"sender": "User", "text": new_query})
        stream_handler.reset()

        def run():
            try:
                agent.invoke(new_query)
            except Exception as e:
                stream_handler.chunks.append(f"[Error: {str(e)}]")
                stream_handler.done = True

        threading.Thread(target=run).start()
        return format_chat_log(history + [{"sender": "Bot", "text": ""}]), history, True, False

    # If streaming in progress → update
    if streaming:
        current_text = stream_handler.get_text()
        temp = history + [{"sender": "Bot", "text": current_text}]

        if stream_handler.is_done():
            history.append({"sender": "Bot", "text": current_text})
            return format_chat_log(history), history, False, True

        return format_chat_log(temp), history, True, False

    # Default
    return format_chat_log(history), history, False, True

# Format messages
def format_chat_log(history):
    return html.Div([
        html.Div([
            html.Strong(f"{msg['sender']}: "),
            html.Span(msg['text'])
        ], style={'marginBottom': '0.5em'}) for msg in history
    ])

if __name__ == "__main__":
    app.run_server(debug=True)
