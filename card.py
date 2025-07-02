class KMArgs(BaseModel):
    time_column: str
    event_column: str
    group_column: str = None

def km_survival_plot(query: str):
    try:
        llm_agent = create_pandas_dataframe_agent(ChatOpenAI(model="gpt-4", temperature=0), df, verbose=False)
        instruction = f"""
        From this dataset, extract the time_column, event_column, and optional group_column for Kaplan-Meier analysis.
        User Query: {query}
        Respond with a JSON object like this:
        {{"time_column": ..., "event_column": ..., "group_column": ...}}. If no group column is mentioned, set it to null.
        """
        json_response = llm_agent.invoke(instruction)
        if isinstance(json_response, str):
            import json
            parsed = KMArgs(**json.loads(json_response))
        else:
            parsed = KMArgs(**json_response)

        T_col = parsed.time_column
        E_col = parsed.event_column
        group_col = parsed.group_column

        if T_col not in df.columns or E_col not in df.columns:
            return f"⚠️ Columns `{T_col}` or `{E_col}` not found."

        fig = go.Figure()
        if group_col and group_col in df.columns:
            for group, subdf in df.groupby(group_col):
                kmf = KaplanMeierFitter()
                kmf.fit(subdf[T_col], event_observed=subdf[E_col], label=str(group))
                fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_[kmf._label], mode="lines", name=str(group)))
            fig.update_layout(title=f"KM Curve by {group_col}", xaxis_title=T_col, yaxis_title="Survival Probability")
        else:
            kmf = KaplanMeierFitter()
            kmf.fit(df[T_col], event_observed=df[E_col], label="All")
            fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_["All"], mode="lines", name="All"))
            fig.update_layout(title="Kaplan-Meier Curve", xaxis_title=T_col, yaxis_title="Survival Probability")

        streamed_plot["fig"] = fig
        return "Kaplan-Meier plot generated."
    except Exception as e:
        return f"❌ KM Error: {e}"



class CoxArgs(BaseModel):
    time_column: str
    event_column: str
    covariates: list[str]

def coxph_plot(query: str):
    try:
        llm_agent = create_pandas_dataframe_agent(ChatOpenAI(model="gpt-4", temperature=0), df, verbose=False)
        instruction = f"""
        From this dataset, extract the time_column, event_column, and covariates list for Cox Proportional Hazards modeling.
        User Query: {query}
        Respond with a JSON object like this:
        {{"time_column": ..., "event_column": ..., "covariates": [...]}}
        """
        import json
        json_response = llm_agent.invoke(instruction)
        if isinstance(json_response, str):
            args = CoxArgs(**json.loads(json_response))
        else:
            args = CoxArgs(**json_response)
        T_col = args.time_column
        E_col = args.event_column
        covs = args.covariates

        for col in [T_col, E_col] + covs:
            if col not in df.columns:
                return f"⚠️ Column `{col}` not found."

        df_cox = df[[T_col, E_col] + covs].dropna()
        df_cox = pd.get_dummies(df_cox, drop_first=True)  # handle categorical

        cph = CoxPHFitter()
        cph.fit(df_cox, duration_col=T_col, event_col=E_col)

        fig = go.Figure()
        summary = cph.summary.reset_index()
        for i, row in summary.iterrows():
            fig.add_trace(go.Bar(
                x=[row["coef"]],
                y=[row["index"]],
                orientation='h',
                error_x=dict(type='data', array=[row["se(coef)"]]),
                name=row["index"]
            ))

        fig.update_layout(title="CoxPH Coefficients", xaxis_title="Coefficient", yaxis_title="Covariate")
        streamed_plot["fig"] = fig
        return "Cox model fitted and coefficients plotted."
    except Exception as e:
        return f"❌ CoxPH Error: {e}"



from lifelines.statistics import logrank_test

class LogRankArgs(BaseModel):
    time_column: str
    event_column: str
    group_column: str

def logrank_comparison(query: str):
    try:
        llm_agent = create_pandas_dataframe_agent(ChatOpenAI(model="gpt-4", temperature=0), df, verbose=False)
        instruction = f"""
        From this dataset, extract the time_column, event_column, and group_column for log-rank test.
        User Query: {query}
        Respond with a JSON object like this:
        {{"time_column": ..., "event_column": ..., "group_column": ...}}
        """
        import json
        json_response = llm_agent.invoke(instruction)
        if isinstance(json_response, str):
            args = LogRankArgs(**json.loads(json_response))
        else:
            args = LogRankArgs(**json_response)

        T_col = args.time_column
        E_col = args.event_column
        G_col = args.group_column

        if G_col not in df.columns:
            return f"⚠️ Group column `{G_col}` not found."

        unique_groups = df[G_col].dropna().unique()
        if len(unique_groups) != 2:
            return f"⚠️ Log-rank test requires exactly 2 groups, found {len(unique_groups)}."

        df1 = df[df[G_col] == unique_groups[0]]
        df2 = df[df[G_col] == unique_groups[1]]

        result = logrank_test(df1[T_col], df2[T_col], event_observed_A=df1[E_col], event_observed_B=df2[E_col])
        p_val = result.p_value
        return f"Log-rank test between {unique_groups[0]} and {unique_groups[1]}: p = {p_val:.4f}"
    except Exception as e:
        return f"❌ Log-rank Error: {e}"

# === Linear Mixed Model Tool using LLM via DataFrame Agent ===
import statsmodels.formula.api as smf

class LMMArgs(BaseModel):
    response: str
    time_column: str
    group_column: str
    subject_column: str

def lmm_plot(query: str):
    try:
        llm_agent = create_pandas_dataframe_agent(ChatOpenAI(model="gpt-4", temperature=0), df, verbose=False)
        instruction = f"""
        From this dataset, extract the response, time_column, group_column, and subject_column for linear mixed model analysis.
        User Query: {query}
        Respond with a JSON object like this:
        {{"response": ..., "time_column": ..., "group_column": ..., "subject_column": ...}}
        """
        import json
        json_response = llm_agent.invoke(instruction)
        if isinstance(json_response, str):
            args = LMMArgs(**json.loads(json_response))
        else:
            args = LMMArgs(**json_response)

        y, time, group, subject = args.response, args.time_column, args.group_column, args.subject_column

        required_cols = [y, time, group, subject]
        if not all(col in df.columns for col in required_cols):
            return f"⚠️ Missing one or more columns: {required_cols}"

        df_clean = df[[y, time, group, subject]].dropna()
        df_clean[group] = df_clean[group].astype(str)

        formula = f"{y} ~ {time} * {group}"
        model = smf.mixedlm(formula, df_clean, groups=df_clean[subject])
        result = model.fit()

        pred_df = df_clean.copy()
        pred_df["predicted"] = result.predict()

        fig = px.line(pred_df, x=time, y="predicted", color=group, title="Smoothed Group-Level Trends with LMM")
        streamed_plot["fig"] = fig

        html_table = html.Div([
            html.H6("📊 LMM Fixed Effects Table"),
            dbc.Table.from_dataframe(result.summary().tables[1], striped=True, bordered=True, hover=True, className="table-sm")
        ])
        streamed_table["html"] = html_table
        return f"Linear mixed model fitted with {y} as response."
    except Exception as e:
        return f"❌ LMM Error: {e}"