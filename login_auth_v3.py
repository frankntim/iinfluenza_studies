import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import sqlite3

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    # Buttons that open modal
    dbc.Button("New Chat", id="new-chat-button", color="primary", className="me-2"),
    dbc.Button("Sidebar New Chat", id="sidebar-new-chat-button", color="secondary"),

    # Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Choose a Model")),
        dbc.ModalBody([
            dcc.Dropdown(
                id='llm-dropdown',
                options=[
                    {'label': 'ChatGPT', 'value': 'ChatGPT'},
                    {'label': 'Gemini', 'value': 'Gemini'},
                    {'label': 'Groq', 'value': 'Groq'}
                ],
                placeholder="Select a model"
            ),
            html.Div(id='warning-message', style={"color": "red", "marginTop": "10px"})
        ])
    ], id='llm-modal', is_open=False)
])

# Callback to open modal from either button
@app.callback(
    Output('llm-modal', 'is_open'),
    Input('new-chat-button', 'n_clicks'),
    Input('sidebar-new-chat-button', 'n_clicks'),
    State('llm-modal', 'is_open'),
    prevent_initial_call=True
)
def open_modal(btn1, btn2, is_open):
    if ctx.triggered_id in ['new-chat-button', 'sidebar-new-chat-button']:
        return True
    return is_open

# Callback to check value and either close modal or show warning
@app.callback(
    Output('llm-modal', 'is_open', allow_duplicate=True),
    Output('warning-message', 'children'),
    Input('llm-dropdown', 'value'),
    prevent_initial_call='initial_duplicate')
def handle_dropdown_selection(value):
    if value:
        conn = sqlite3.connect('permission_db.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM LLMtypes WHERE name = ?", (value,))
        result = cursor.fetchone()
        conn.close()

        if result:
            return False, ""
        else:
            return True, "LLM model not found"

    return dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
