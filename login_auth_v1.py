# ... (existing import section)
import sqlite3  # <--- ensure this is present

# Inside your app.layout, add this at the end:
app.layout = html.Div([
    # ... existing layout components ...

    # Modal popup for dropdown selection
    dbc.Modal(
        [
            dbc.ModalHeader("Select an LLM"),
            dbc.ModalBody([
                dcc.Dropdown(
                    id="popup-dropdown",
                    options=[
                        {"label": "ChatGPT", "value": "ChatGPT"},
                        {"label": "Gemini", "value": "Gemini"},
                        {"label": "Groq", "value": "Groq"},
                    ],
                    placeholder="Choose a model",
                ),
                html.Div(id="dropdown-warning", style={"color": "red", "marginTop": "10px"})
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="modal-close-button", color="danger")
            ),
        ],
        id="popup-modal",
        is_open=False,
        centered=True,
    ),

    # Overlay to simulate disabling app
    html.Div(id="app-overlay", className="app-overlay hidden"),

    # Store
    dcc.Store(id="app-disabled", data=False)
])
#################################
.app-overlay {
  position: fixed;
  top: 0; left: 0;
  width: 100vw; height: 100vh;
  background-color: rgba(255, 255, 255, 0.8);
  z-index: 9999;
  display: flex;
  justify-content: center;
  align-items: center;
  font-size: 24px;
  font-weight: bold;
  color: red;
}
.app-overlay.hidden {
  display: none;
}


###################
# Toggle modal open or close based on button or dropdown
@app.callback(
    [Output("popup-modal", "is_open"),
     Output("dropdown-warning", "children")],
    [Input("new-chat-button", "n_clicks"),
     Input("popup-dropdown", "value")],
    [State("popup-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_and_validate_modal(n_clicks, dropdown_value, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "new-chat-button":
        return True, ""
    elif trigger == "popup-dropdown" and dropdown_value:
        conn = sqlite3.connect("permission.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM LLMtypes WHERE name = ?", (dropdown_value,))
        count = cursor.fetchone()[0]
        conn.close()

        if count == 0:
            return True, "This item is not available"
        else:
            return False, ""

    return is_open, ""

# Disable app when modal is closed with button
@app.callback(
    Output("app-overlay", "className"),
    Input("modal-close-button", "n_clicks"),
    prevent_initial_call=True
)
def disable_app(n_clicks):
    if n_clicks:
        return "app-overlay"
    return "app-overlay hidden"
