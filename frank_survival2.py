import plotly.graph_objects as go
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import pandas as pd
import numpy as np
from dash import Dash, dcc, html
import pandas as pd
from lifelines.datasets import load_rossi



# 1. Cox Fitted Survival Curve
def plot_cox_fitted(df, duration_col, event_col, covariates, timeline=None):
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
def plot_partial_effect(df, duration_col, event_col, covariates, target_covariate, values):
    cph = CoxPHFitter()
    cph.fit(df[[duration_col, event_col] + covariates], duration_col=duration_col, event_col=event_col)

    base_df = df[covariates].median().to_frame().T
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


df = load_rossi()

app = Dash(__name__)

cox_fig = plot_cox_fitted(df, duration_col='week', event_col='arrest', covariates=['age', 'fin', 'prio'])
logrank_fig = plot_logrank(df, duration_col='week', event_col='arrest', group_col='fin')
partial_fig = plot_partial_effect(df, duration_col='week', event_col='arrest',
                                  covariates=['age', 'fin', 'prio'], target_covariate='age',
                                  values=[20, 30, 40, 50])

app.layout = html.Div([
    html.H2("Cox Fitted Plot"),
    dcc.Graph(figure=cox_fig),

    html.H2("Log-Rank Plot"),
    dcc.Graph(figure=logrank_fig),

    html.H2("Partial Effect on Outcome"),
    dcc.Graph(figure=partial_fig),
])

if __name__ == '__main__':
    app.run_server(debug=True)