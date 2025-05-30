from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Assume these are your existing figures
# (replace with your own fig1 and fig2)
import plotly.express as px
df = px.data.iris()
fig1 = px.scatter(df, x="sepal_width", y="sepal_length", color="species", title="Sepal Dimensions")
fig2 = px.line(df, x="petal_width", y="petal_length", color="species", title="Petal Dimensions")

# Create subplot layout (1 row, 2 columns for side-by-side; use rows=2, cols=1 for stacked)
combined_fig = make_subplots(rows=1, cols=2, subplot_titles=(fig1.layout.title.text, fig2.layout.title.text))

# Add traces from fig1 to subplot 1
for trace in fig1.data:
    combined_fig.add_trace(trace, row=1, col=1)

# Add traces from fig2 to subplot 2
for trace in fig2.data:
    combined_fig.add_trace(trace, row=1, col=2)

# Update layout
combined_fig.update_layout(
    height=600, width=1000,
    title_text="Combined Subplots of Two Figures",
    showlegend=False  # If you want to disable duplicate legends
)

combined_fig.show()




import pandas as pd
import plotly.graph_objects as go
from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter

def cox_summary_plot(data, duration_col, event_col):
    cph = CoxPHFitter()
    cph.fit(data, duration_col=duration_col, event_col=event_col)
    summary_df = cph.summary.reset_index().round(3)

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(summary_df.columns),
                    fill_color='lightgrey',
                    align='left'),
        cells=dict(values=[summary_df[col] for col in summary_df.columns],
                   fill_color='white',
                   align='left'))
    ])
    
    fig.update_layout(title='Cox PH Model Summary', margin=dict(l=0, r=0, t=40, b=0))
    return fig

# Example usage:
df = load_rossi()
fig = cox_summary_plot(df, duration_col='week', event_col='arrest')
fig.show()
