import dash
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import dash_bootstrap_components as dbc
import os
import json
import plotly.express as px

# Placeholder for the custom callback handler (Requires additional implementation)
class StreamingCallbackHandler:
    def __init__(self):
        self.response_tokens = []
        self.chart_data = None  # To store chart information if needed

    def on_llm_new_token(self, token: str, **kwargs):
        self.response_tokens.append(token)

    def on_agent_action(self, action, **kwargs):
        # Handle agent actions (e.g., tool calls)
        pass

    def on_agent_finish(self, finish_state, **kwargs):
        # Process the final response and check for chart instructions
        try:
            response_json = json.loads("".join(self.response_tokens))  # Assume agent provides JSON
            if "bar_chart" in response_json:
                self.chart_data = response_json["bar_chart"]
        except json.JSONDecodeError:
            pass  # Handle non-JSON responses

# Load data (replace with your actual data loading)
try:
    df = pd.read_csv('titanic.csv')
except FileNotFoundError:
    print("Error: titanic.csv not found.")
    exit()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initialize the LangChain agent
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key="YOUR_OPENAI_API_KEY", streaming=True)
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True
)

# Modal component for the chatbot
modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Titanic Chatbot")),
        dbc.ModalBody(id="modal-body"),
        dbc.ModalFooter(
            [
                dbc.Input(id="user-input", type="text", placeholder="Enter your query..."),
                dbc.Button("Send", id="send-button", className="ms-auto", n_clicks=0),
                dbc.Button("Close", id="close-modal", className="ms-auto", n_clicks=0),
            ]
        ),
    ],
    id="modal",
    is_open=False,
    size="xl",  # Set modal size to "xl" for better chart display
)

app.layout = html.Div(
    [
        html.H1("Titanic Data Analysis with AI Agent"),
        dbc.Button("Open Chatbot", id="open-modal", n_clicks=0),
        modal,
        dcc.Interval(id='interval-component', interval=100, n_intervals=0, disabled=True), # Used for background streaming
        dcc.Store(id='streaming-data', data={}), # To store streaming data
    ]
)

@app.callback(
    Output("modal", "is_open"),
    [Input("open-modal", "n_clicks"), Input("close-modal", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        return not is_open
    return is_open

# Callback to send the query and trigger streaming
@app.callback(
    Output("interval-component", "disabled"),
    Input("send-button", "n_clicks"),
    State("user-input", "value"),
)
def start_agent_streaming(n_clicks, query):
    if n_clicks > 0 and query:
        # Trigger the agent run in a background task here
        # The background task would use the StreamingCallbackHandler to capture tokens
        # and send them to the frontend (e.g., via dcc.Store or a different mechanism)
        return False # Enable the interval component for streaming updates
    return True

# Callback to update modal body with streamed data
@app.callback(
    Output("modal-body", "children"),
    Input("interval-component", "n_intervals"),
    State("streaming-data", "data"),
)
def update_modal_body(n_intervals, streaming_data):
    if streaming_data:
        response_tokens = streaming_data.get("tokens", [])
        chart_data = streaming_data.get("chart_data")

        if chart_data:
            # Create and display the Plotly chart
            try:
                fig = px.bar(df, x=chart_data["x_col"], y=chart_data["y_col"], title=chart_data.get("title", "Bar Chart"))
                return [html.Div("".join(response_tokens)), dcc.Graph(figure=fig)]
            except KeyError:
                return [html.Div("Error creating chart.")]
        else:
            # Display the streamed text response
            return [html.Div("".join(response_tokens))]
    return []

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
