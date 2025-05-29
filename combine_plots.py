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
