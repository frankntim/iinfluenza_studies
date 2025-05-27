import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from lifelines import CoxPHFitter

# Sample data - replace with your actual dataframe
df = pd.DataFrame({
    'duration': [5, 10, 15, 20, 25],
    'event': [1, 0, 1, 0, 1],
    'age': [30, 40, 50, 60, 70],
    'bmi': [22, 25, 27, 30, 35]
})

# Fit Cox model
cph = CoxPHFitter()
cph.fit(df, duration_col='duration', event_col='event')

def plot_cox_summary(cph):
    """Returns a Plotly figure of CoxPH model summary"""
    summary = cph.summary.reset_index()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=summary['coef'],
        y=summary['covariate'],
        error_x=dict(
            type='data',
            symmetric=False,
            array=summary['upper 95%'] - summary['coef'],
            arrayminus=summary['coef'] - summary['lower 95%']
        ),
        mode='markers',
        marker=dict(color='blue', size=10),
    ))

    fig.update_layout(title='CoxPH Coefficients with 95% CI',
                      xaxis_title='Coefficient',
                      yaxis_title='Covariates')
    return fig

def plot_partial_effects(cph, covariates):
    """Returns Plotly figures for each covariate's partial effect on survival"""
    figures = []
    for cov in covariates:
        values = sorted(df[cov].unique())
        fig = go.Figure()
        for val in values:
            surv = cph.predict_survival_function({cov: val}, times=df['duration'])
            fig.add_trace(go.Scatter(x=surv.index, y=surv.values.flatten(), name=f'{cov}={val}'))
        fig.update_layout(title=f'Partial Effect of {cov} on Survival',
                          xaxis_title='Time',
                          yaxis_title='Survival Probability')
        figures.append(fig)
    return figures

# Dash App Layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    html.H2("Cox Proportional Hazards Model"),
    dcc.Graph(id='cox-summary', figure=plot_cox_summary(cph)),
    html.H4("Partial Effects of Covariates"),
    dcc.Dropdown(id='covariate-dropdown',
                 options=[{'label': c, 'value': c} for c in ['age', 'bmi']],
                 multi=True,
                 value=['age']),
    html.Div(id='partial-effect-plots')
])

@app.callback(
    Output('partial-effect-plots', 'children'),
    Input('covariate-dropdown', 'value')
)
def update_partial_effect_plots(selected_covariates):
    figures = plot_partial_effects(cph, selected_covariates)
    return [dcc.Graph(figure=fig) for fig in figures]

if __name__ == '__main__':
    app.run_server(debug=True)
