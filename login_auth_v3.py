import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import sqlite3

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Function to get allowed LLMs from the database
def fetch_allowed_llms():
    conn = sqlite3.connect('permission_db.db')
    cursor = conn.cursor()
    cursor.execute("SELECT llm_name FROM LLMtypes")
    llms = [row[0] for row in cursor.fetchall()]
    conn.close()
    return llms

# List of dropdown options
llm_options = ['ChatGPT', 'Groq', 'Gemini']

# Layout
app.layout = html.Div([
    dcc.Store(id="popup-dismissed", data=False),

    # Popup modal
    dbc.Modal([
        dbc.ModalHeader("User Login"),
        dbc.ModalBody([
            dbc.Input(id="username", placeholder="Enter your username", type="text", className="mb-3"),
            dbc.Label("Select LLM Type"),
            dcc.Dropdown(
                id='llm-dropdown',
                options=[{'label': llm, 'value': llm} for llm in llm_options],
                placeholder="Select LLM type"
            ),
            html.Div(id="login-error", className="text-danger mt-2")
        ]),
    ],
    id="login-popup",
    is_open=True,
    backdrop='static',
    keyboard=False
    ),

    html.Div(id='app-content', children=[
        html.H3("Welcome to the LLM Dashboard"),
        html.P("This content is visible after successful login.")
    ])
])

# Callback to auto-close popup based on valid LLM selection
@app.callback(
    Output("login-popup", "is_open"),
    Output("login-error", "children"),
    Input("llm-dropdown", "value"),
    prevent_initial_call=True
)
def handle_llm_selection(selected_llm):
    if not selected_llm:
        return True, ""
    
    allowed_llms = fetch_allowed_llms()
    if selected_llm in allowed_llms:
        return False, ""
    else:
        return True, f"Access denied: {selected_llm} is not permitted."

if __name__ == '__main__':
    app.run_server(debug=True)
