import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from typing import Dict
from openai import OpenAI
import os


# ================== CONFIG ====================
DIMENSIONS = ["Correctness", "Relevance", "Completeness", "Conciseness"]
SCORE_RANGE = (0, 5)

# ============= LLM-AS-JUDGE FUNCTION ===========
def llm_multi_metric_evaluation(expected: str, response: str, model="gpt-4o-mini") -> Dict:
    client = OpenAI()

    system_prompt = f"""You are an impartial evaluator.
Score the model's answer against the expected answer on these dimensions:
- Correctness (factual accuracy)
- Relevance (answers the actual question)
- Completeness (covers key points)
- Conciseness (clear, not verbose)

For each dimension, return an integer score from {SCORE_RANGE[0]} to {SCORE_RANGE[1]}.
Provide a short justification.

Respond with strict JSON:
{{
 "{DIMENSIONS[0]}": {{"score": int, "justification": str}},
 "{DIMENSIONS[1]}": {{"score": int, "justification": str}},
 "{DIMENSIONS[2]}": {{"score": int, "justification": str}},
 "{DIMENSIONS[3]}": {{"score": int, "justification": str}}
}}"""

    user_prompt = f"Expected Answer:\n{expected}\n\nModel Response:\n{response}\n\nEvaluate now."

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    return json.loads(completion.choices[0].message.content)


# ============= PLOTTING HELPERS ================
def radar_plot(categories, values, title):
    r = values + [values[0]]
    t = categories + [categories[0]]
    fig = go.Figure(
        data=go.Scatterpolar(r=r, theta=t, fill="toself", name="Mean")
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=SCORE_RANGE)),
        showlegend=False,
        title=title
    )
    return fig


# ============= STREAMLIT APP ===================
st.title("📊 LLM Multi-Metric Evaluation Dashboard")

st.write("Upload a CSV file with columns: `question, expected, response, model_name`")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of dataset")
    st.dataframe(df.head())

    results = []
    for idx, row in df.iterrows():
        eval_result = llm_multi_metric_evaluation(row["expected"], row["response"])
        for dim in DIMENSIONS:
            results.append({
                "row_index": idx,
                "question": row["question"],
                "model_name": row["model_name"],
                "dimension": dim,
                "score": eval_result[dim]["score"],
                "justification": eval_result[dim]["justification"],
            })

    results_df = pd.DataFrame(results)
    st.write("### Evaluation Results")
    st.dataframe(results_df)

    # Overall aggregate
    overall = results_df.groupby("dimension")["score"].mean().reset_index()

    # Radar Plot
    st.write("### Radar Plot (Overall Mean Scores)")
    st.plotly_chart(radar_plot(overall["dimension"].tolist(),
                               overall["score"].tolist(),
                               "Overall Mean Scores"), use_container_width=True)

    # Bar Plot
    st.write("### Bar Plot (Overall Mean Scores)")
    fig_bar = px.bar(overall, x="dimension", y="score", text=overall["score"].round(2),
                     range_y=SCORE_RANGE, title="Overall Mean Scores")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Box Plot (distribution of scores)
    st.write("### Box Plot (Score Distribution by Dimension)")
    fig_box = px.box(results_df, x="dimension", y="score", points="all",
                     range_y=SCORE_RANGE, title="Score Distribution")
    st.plotly_chart(fig_box, use_container_width=True)

    # Heatmap by model_name
    st.write("### Heatmap by Model and Dimension")
    heat = results_df.groupby(["model_name", "dimension"])["score"].mean().reset_index()
    fig_heat = px.density_heatmap(heat, x="dimension", y="model_name", z="score",
                                  histfunc="avg", nbinsx=len(DIMENSIONS),
                                  title="Mean Scores by Model and Dimension")
    st.plotly_chart(fig_heat, use_container_width=True)

    # Grouped Bar
    st.write("### Grouped Bar by Model and Dimension")
    fig_grouped = px.bar(heat, x="dimension", y="score", color="model_name",
                         barmode="group", range_y=SCORE_RANGE,
                         title="Scores by Model and Dimension")
    st.plotly_chart(fig_grouped, use_container_width=True)
