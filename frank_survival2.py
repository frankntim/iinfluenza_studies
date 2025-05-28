import plotly.graph_objects as go
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import pandas as pd
import numpy as np
from dash import Dash, dcc, html

from lifelines.datasets import load_rossi
df = load_rossi()
app = Dash(__name__)

def get_covariates(df, duration_col, event_col, exclude=[]):
    """Helper to get covariate columns automatically"""
    return [col for col in df.columns if col not in [duration_col, event_col] + exclude]

# 1. Cox Fitted Survival Curve
def plot_cox_fitted(df, duration_col, event_col, timeline=None):
    covariates = get_covariates(df, duration_col, event_col)
    cph = CoxPHFitter()
    cph.fit(df[[duration_col, event_col] + covariates], duration_col=duration_col, event_col=event_col)

    if timeline is None:
        timeline = np.linspace(0, df[duration_col].max(), 100)

    surv_df = cph.predict_survival_function(df[covariates], times=timeline)
    
    fig = go.Figure()
    for i in range(surv_df.shape[1]):
        fig.add_trace(go.Scatter(x=timeline, y=surv_df.iloc[:, i], mode='lines', name=f'Sample {i+1}'))

    fig.update_layout(title='Cox Fitted Survival Curves',
                      xaxis_title='Time',
                      yaxis_title='Survival Probability')
    return fig

# 2. Log-rank Test Plot
def plot_logrank(df, duration_col, event_col, group_col):
    fig = go.Figure()
    kmf = KaplanMeierFitter()
    
    for name, grouped_df in df.groupby(group_col):
        kmf.fit(grouped_df[duration_col], grouped_df[event_col])
        fig.add_trace(go.Scatter(
            x=kmf.survival_function_.index,
            y=kmf.survival_function_['KM_estimate'],
            mode='lines',
            name=str(name)
        ))

    groups = df[group_col].unique()
    if len(groups) == 2:
        g1 = df[df[group_col] == groups[0]]
        g2 = df[df[group_col] == groups[1]]
        result = logrank_test(g1[duration_col], g2[duration_col], g1[event_col], g2[event_col])
        fig.update_layout(title=f'Log-Rank Test p-value: {result.p_value:.4f}')
    else:
        fig.update_layout(title='Log-Rank Plot')

    fig.update_layout(xaxis_title='Time', yaxis_title='Survival Probability')
    return fig

# 3. Partial Effect on Outcome
def plot_partial_effect(df, duration_col, event_col, target_covariate, values):
    covariates = get_covariates(df, duration_col, event_col, exclude=[target_covariate])
    cph = CoxPHFitter()
    cph.fit(df[[duration_col, event_col] + covariates + [target_covariate]],
            duration_col=duration_col, event_col=event_col)

    base_df = df[covariates + [target_covariate]].median().to_frame().T
    fig = go.Figure()
    timeline = np.linspace(0, df[duration_col].max(), 100)

    for val in values:
        temp_df = base_df.copy()
        temp_df[target_covariate] = val
        surv = cph.predict_survival_function(temp_df, times=timeline)
        fig.add_trace(go.Scatter(x=timeline, y=surv.iloc[:, 0], mode='lines', name=f'{target_covariate}={val}'))

    fig.update_layout(title=f'Partial Effect of {target_covariate} on Survival',
                      xaxis_title='Time',
                      yaxis_title='Survival Probability')
    return fig


def plot_cox_coefficients(df, duration_col, event_col):
    covariates = [col for col in df.columns if col not in [duration_col, event_col]]
    
    cph = CoxPHFitter()
    cph.fit(df[[duration_col, event_col] + covariates], duration_col=duration_col, event_col=event_col)
    
    summary = cph.summary.reset_index()
    summary['HR'] = np.exp(summary['coef'])
    summary['CI_lower'] = np.exp(summary['coef lower 95%'])
    summary['CI_upper'] = np.exp(summary['coef upper 95%'])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=summary['HR'],
       # y=summary['index'],
        y=summary.index,
        mode='markers',
        error_x=dict(
            type='data',
            symmetric=False,
            array=summary['CI_upper'] - summary['HR'],
            arrayminus=summary['HR'] - summary['CI_lower']
        ),
        marker=dict(size=10, color='blue'),
        name='Hazard Ratio'
    ))

    fig.add_shape(type='line', x0=1, x1=1, y0=-0.5, y1=len(summary)-0.5,
                  line=dict(color='red', dash='dash'))

    fig.update_layout(
        title="Cox Model Coefficients (Hazard Ratios)",
        xaxis_title="Hazard Ratio (log scale)",
        yaxis_title="Covariate",
        xaxis_type="log",
        height=400 + len(summary)*20
    )

    return fig





cox_fig = plot_cox_fitted(df, duration_col='week', event_col='arrest')
logrank_fig = plot_logrank(df, duration_col='week', event_col='arrest', group_col='fin')
partial_fig = plot_partial_effect(df, duration_col='week', event_col='arrest',
                                  target_covariate='age', values=[20, 30, 40, 50])
coef_fig = plot_cox_coefficients(df, duration_col='week', event_col='arrest')

app.layout = html.Div([
    html.H2("Cox Fitted Plot"),
    dcc.Graph(figure=cox_fig),

    html.H2("Log-Rank Plot"),
    dcc.Graph(figure=logrank_fig),

    html.H2("Partial Effect on Outcome"),
    dcc.Graph(figure=partial_fig),

    html.H2("Cox Fitted Coefficient"),
    dcc.Graph(figure=coef_fig),
])

if __name__ == '__main__':
    app.run_server(debug=True)