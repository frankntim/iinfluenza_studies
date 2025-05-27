import io
import base64
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

# Sample DataFrame
df = pd.DataFrame({
    'duration': [5, 10, 15, 20, 25],
    'event': [1, 0, 1, 0, 1],
    'age': [30, 40, 50, 60, 70],
    'bmi': [22, 25, 27, 30, 35]
})

# Fit CoxPH model
cph = CoxPHFitter()
cph.fit(df, duration_col='duration', event_col='event')

def fig_to_base64(fig):
    """Convert a Matplotlib figure to base64 image for Dash"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f'data:image/png;base64,{img_base64}'

def get_cox_plot_img():
    fig = cph.plot()
    return fig_to_base64(fig.figure)

def get_partial_effect_img(covariates):
    fig, ax = plt.subplots()
    for cov in covariates:
        if cov in df.columns:
            try:
                cph.plot_partial_effects_on_outcome(cov, values=df[cov].unique(), ax=ax)
            except:
                continue
    return fig_to_base64(fig)

# Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    html.H2("Cox Model Summary"),
    html.Img(id='cox-summary-img', src=get_cox_plot_img(), style={'width': '100%'}),
    html.H4("Partial Effects on Outcome"),
    dcc.Dropdown(id='covariate-dropdown',
                 options=[{'label': c, 'value': c} for c in ['age', 'bmi']],
                 value=['age'],
                 multi=True),
    html.Img(id='partial-effects-img', style={'width': '100%'})
], fluid=True)

@app.callback(
    Output('partial-effects-img', 'src'),
    Input('covariate-dropdown', 'value')
)
def update_partial_effect_img(selected_covs):
    return get_partial_effect_img(selected_covs)

if __name__ == '__main__':
    app.run_server(debug=True)
