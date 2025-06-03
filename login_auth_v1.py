dbc.Modal(
    [
        dbc.ModalHeader("Select an Item"),
        dbc.ModalBody(
            dcc.Dropdown(
                id="popup-dropdown",
                options=[
                    {"label": "Option 1", "value": "opt1"},
                    {"label": "Option 2", "value": "opt2"},
                    {"label": "Option 3", "value": "opt3"},
                ],
                placeholder="Choose an option",
            )
        ),
    ],
    id="popup-modal",
    is_open=False,
    centered=True,
),


###################
@app.callback(
    Output("popup-modal", "is_open"),
    [Input("new-chat-button", "n_clicks"),
     Input("popup-dropdown", "value")],
    [State("popup-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_modal(open_clicks, dropdown_value, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "new-chat-button":
        return True  # open modal
    elif triggered_id == "popup-dropdown" and dropdown_value:
        return False  # close modal on selection

    return is_open
