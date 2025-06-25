import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI  # Using ChatOpenAI as requested
import dash_bootstrap_components as dbc
import os
import io

# Load the Titanic dataset
try:
    df = pd.read_csv('titanic.csv')  # Make sure titanic.csv is in the same directory
except FileNotFoundError:
    print("Error: titanic.csv not found. Please ensure the file is in the same directory.")
    exit()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initialize the LangChain agent
# Replace "YOUR_OPENAI_API_KEY" with your actual OpenAI API key
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key="YOUR_OPENAI_API_KEY", streaming=True) # Set streaming=True for ChatOpenAI
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True # Note: Use allow_dangerous_code=True with caution in production
)

# Modal component for the chatbot
modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Titanic Chatbot")),
        dbc.ModalBody(id="modal-body"),  # This is where the response will be displayed
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
)

# App layout
app.layout = html.Div(
    [
        html.H1("Titanic Data Analysis with AI Agent"),
        dbc.Button("Open Chatbot", id="open-modal", n_clicks=0),
        modal,
    ]
)

# Callback to open/close the modal
@app.callback(
    Output("modal", "is_open"),
    [Input("open-modal", "n_clicks"), Input("close-modal", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        return not is_open
    return is_open

# Callback to send the query and stream the response
@app.callback(
    Output("modal-body", "children"),  # Output to the modal body
    Input("send-button", "n_clicks"),
    State("user-input", "value"),
)
def send_query_and_stream_response(n_clicks, query):
    if n_clicks > 0 and query:
        try:
            # Running the agent and capturing the output
            # Note: For true token-by-token streaming, more advanced techniques
            # like callbacks or custom runnables would be needed.
            response = agent.run(query)

            # Display the final response in the modal body
            return [html.P(response)]

        except Exception as e:
            return [html.P(f"Error: {e}")]

    return []

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
