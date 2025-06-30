app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    style={
        "backgroundImage": "url('/assets/topography.png')",
        "backgroundRepeat": "repeat",
        "backgroundSize": "auto",
        "minHeight": "100vh",
        "padding": "2rem"
    },
    children=[
        dbc.Card(
            dbc.CardBody([
                html.H4("Welcome to the Survival Analysis Chatbot", className="card-title"),
                html.P("Click below to open the analysis assistant.", className="card-text"),
                dbc.Button("Open Chatbox", id="open-modal", color="primary")
            ]),
            style={"maxWidth": "500px", "margin": "auto", "marginTop": "4rem", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"}
        ),

        dbc.Modal(
            [
                dbc.ModalHeader("Chat with Data Agent"),
                dbc.ModalBody(
                    [
                        html.Div(id="chat-log", style={
                            "whiteSpace": "pre-wrap",
                            "overflowY": "auto",
                            "height": "300px",
                            "padding": "1rem",
                            "backgroundColor": "rgba(255,255,255,0.85)",
                            "borderRadius": "10px"
                        }),
                        dcc.Graph(id="chart-output", style={"marginTop": "1rem"}),
                        html.Div(id="table-output", style={
                            "marginTop": "1rem",
                            "backgroundColor": "rgba(255,255,255,0.85)",
                            "padding": "1rem",
                            "borderRadius": "10px"
                        }),
                        dbc.Input(id="user-input", placeholder="Ask a question...", type="text"),
                        dbc.Button("Submit", id="submit-btn", color="success", className="mt-2")
                    ]
                ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-modal", className="ml-auto")
                )
            ],
            id="chat-modal",
            is_open=False,
            size="xl",
            scrollable=True,
            backdrop="static"
        )
    ]
)
