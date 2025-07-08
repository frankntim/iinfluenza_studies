# Databricks ipywidgets notebook: Data Catalog & Table Explorer

# COMMAND ----------
# MAGIC %md
# # Data Catalog & Table Explorer
# This notebook provides an interactive UI (powered by `ipywidgets`) to browse Unity Catalog catalogs, schemas, and tables, and to run ad‑hoc SQL queries directly from a Databricks notebook.

# COMMAND ----------
# Install ipywidgets if needed (Databricks Runtime 14+ comes with it)
# Uncomment the line below if ipywidgets is missing in your cluster environment
# %pip install --quiet ipywidgets

# COMMAND ----------
import ipywidgets as widgets
from IPython.display import display, clear_output
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def list_catalogs() -> list[str]:
    """Return a list of catalog names visible to the current user."""
    return [row.catalog_name for row in spark.sql("SHOW CATALOGS").collect()]


def list_schemas(catalog: str) -> list[str]:
    """Return a list of schemas/databases for a given catalog."""
    return [row.databaseName for row in spark.sql(f"SHOW SCHEMAS IN {catalog}").collect()]


def list_tables(catalog: str, schema: str) -> list[str]:
    """Return a list of tables for a given catalog & schema."""
    rows = spark.sql(f"SHOW TABLES IN {catalog}.{schema}").collect()
    return [row.tableName for row in rows]

# -----------------------------------------------------------------------------
# Widgets definitions
# -----------------------------------------------------------------------------

catalog_dropdown = widgets.Dropdown(
    options=list_catalogs(),
    description="Catalog:",
    layout=widgets.Layout(width="300px"),
)

schema_dropdown = widgets.Dropdown(
    options=[],
    description="Schema:",
    layout=widgets.Layout(width="300px"),
)

tables_select = widgets.SelectMultiple(
    options=[],
    description="Tables:",
    rows=8,
    layout=widgets.Layout(width="300px"),
)

limit_int = widgets.BoundedIntText(
    value=20,
    min=1,
    max=10000,
    step=1,
    description="Show rows:",
    layout=widgets.Layout(width="200px"),
)

run_button = widgets.Button(
    description="Run Query",
    button_style="success",
    tooltip="Run the generated query and display the result",
)

custom_sql = widgets.Textarea(
    placeholder="Or write your own SQL here…",
    description="SQL:",
    layout=widgets.Layout(width="600px", height="120px"),
)

output = widgets.Output()

# -----------------------------------------------------------------------------
# Callback functions
# -----------------------------------------------------------------------------

def on_catalog_change(change):
    catalog = change["new"]
    schema_dropdown.options = list_schemas(catalog)
    # Automatically trigger schema update for the first schema in list
    if schema_dropdown.options:
        schema_dropdown.value = schema_dropdown.options[0]


def on_schema_change(change):
    catalog = catalog_dropdown.value
    schema = change["new"]
    tables_select.options = list_tables(catalog, schema)


def run_query(_):
    with output:
        clear_output()
        if custom_sql.value.strip():
            sql = custom_sql.value.strip()
            print("Running custom SQL:")
            print(sql)
        else:
            selected = list(tables_select.value)
            if not selected:
                print("⚠️ Please select at least one table or write custom SQL.")
                return
            catalog = catalog_dropdown.value
            schema = schema_dropdown.value
            table_refs = [f"{catalog}.{schema}.{t}" for t in selected]
            sql = "SELECT * FROM " + ", ".join(table_refs) + f" LIMIT {limit_int.value}"
            print("Running generated SQL:")
            print(sql)
        try:
            df = spark.sql(sql)
            display(df.limit(limit_int.value).toPandas())
        except Exception as e:
            print("Query failed:", e)

# -----------------------------------------------------------------------------
# Widget event bindings
# -----------------------------------------------------------------------------

catalog_dropdown.observe(on_catalog_change, names="value")
schema_dropdown.observe(on_schema_change, names="value")
run_button.on_click(run_query)

# Trigger initial population of schema & table lists
on_catalog_change({"new": catalog_dropdown.value})

# -----------------------------------------------------------------------------
# Layout & display
# -----------------------------------------------------------------------------

ui = widgets.VBox([
    widgets.HBox([catalog_dropdown, schema_dropdown]),
    widgets.HBox([tables_select, widgets.VBox([limit_int, run_button])]),
    custom_sql,
    output,
])

display(ui)

# COMMAND ----------
# MAGIC %md
# ### Tips
# * **Generated query** — If you select one or multiple tables but leave the SQL box blank, the notebook builds a simple `SELECT *` query for you.
# * **Custom query** — Enter any valid SQL in the text area to run it directly. The generated query is ignored if the box is not empty.
# * **Row limit** — Adjust the slider to control how many rows are displayed from the query result.
# * **Multiple tables** — Hold *⌘/Ctrl* to select more than one table in the list.
# * **Schema refresh** — If new catalogs or tables are created during your session, re‑run the first helper cell to refresh the widget options.

# Enjoy your interactive catalog explorer!
