...

# === Global Store ===
df = pd.read_csv("titanic.csv")
streamed_plot = {}
streamed_table = {}

# === Upload Callback ===
@app.callback(
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)
def upload_csv(contents, filename):
    import base64
    import io
    global df
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return f"✅ Uploaded {filename} with {df.shape[0]} rows."

# === Add Upload to Card Body ===
dbc.CardBody([
    html.H5("General Analysis", className="card-title"),
    dcc.Upload(
        id="upload-data",
        children=html.Div(["📁 Drag and Drop or ", html.A("Select CSV")]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
            'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center'
        },
        multiple=False
    ),
    html.Div(id="upload-status", style={"marginTop": "10px", "color": "green"})
])

# Apply similar structure to all four cards by updating each dbc.CardBody similarly.

if __name__ == "__main__":
    app.run_server(debug=True)
