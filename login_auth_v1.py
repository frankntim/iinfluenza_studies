import dash
from dash import html, Input, Output, State, dcc
import dash_bootstrap_components as dbc
from flask import Flask
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required

# -----------------------------
# Flask Setup with Flask-Login
# -----------------------------
server = Flask(__name__)
server.secret_key = 'supersecretkey'  # For session cookies

login_manager = LoginManager()
login_manager.init_app(server)

# Dummy user store
USER_DB = {"admin": "password123"}

# User model
class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(user_id):
    if user_id in USER_DB:
        return User(user_id)
    return None

# -----------------------------
# Dash App
# -----------------------------
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Secure Dash App"
app.config.suppress_callback_exceptions = True

# -----------------------------
# Login Modal
# -----------------------------
login_modal = dbc.Modal(
    [
        dbc.ModalHeader("Login Required"),
        dbc.ModalBody([
            dbc.Input(id="username", placeholder="Enter username", type="text", className="mb-2"),
            dbc.Input(id="password", placeholder="Enter password", type="password", className="mb-2"),
            html.Div(id='login-feedback', className='text-danger'),
        ]),
        dbc.ModalFooter(
            dbc.Button("Login", id="login-button", className="ms-auto", n_clicks=0)
        ),
    ],
    id="login-modal",
    is_open=True,  # Open by default
    backdrop='static',
    keyboard=False,
)

# -----------------------------
# App Layout
# -----------------------------
app.layout = html.Div([
    dcc.Location(id='url'),
    dcc.Store(id='login-state', storage_type='session'),
    login_modal,
    html.Div(id="main-content")
])

# -----------------------------
# Main Page Content
# -----------------------------
def layout_main_page(username):
    return html.Div([
        html.H3(f"Welcome, {username}! You are logged in."),
        html.Button("Logout", id="logout-button", className="btn btn-danger mt-3"),
    ])

# -----------------------------
# Callback: Show/Hide Modal
# -----------------------------
@app.callback(
    Output("login-modal", "is_open"),
    Output("main-content", "children"),
    Output("login-feedback", "children"),
    Output("login-state", "data"),
    Input("login-button", "n_clicks"),
    State("username", "value"),
    State("password", "value"),
    prevent_initial_call=True
)
def handle_login(n_clicks, username, password):
    if username in USER_DB and USER_DB[username] == password:
        user = User(username)
        login_user(user)  # Flask-login session
        return False, layout_main_page(username), "", True
    else:
        return True, "", "Invalid credentials.", False

# -----------------------------
# Callback: Check Session on Load
# -----------------------------
@app.callback(
    Output("main-content", "children", allow_duplicate=True),
    Output("login-modal", "is_open", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call='initial_duplicate'
)
def check_session(pathname):
    if current_user.is_authenticated:
        return layout_main_page(current_user.id), False
    else:
        return "", True

# -----------------------------
# Callback: Logout
# -----------------------------
@app.callback(
    Output("main-content", "children", allow_duplicate=True),
    Output("login-modal", "is_open", allow_duplicate=True),
    Input("logout-button", "n_clicks"),
    prevent_initial_call=True
)
def logout(n):
    logout_user()
    return "", True

# -----------------------------
# Run the App
# -----------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
