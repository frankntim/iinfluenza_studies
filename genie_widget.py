"""
Databricks Catalog Browser (ipywidgets)

A reusable Python module that spins up an interactive Unity Catalog explorer inside a Databricks notebook.

Quick start (**notebook cell**):

```python
# (1) Ensure the script is on DBFS or workspace, then load it
%run ./databricks_catalog_browser.py  # path may vary

# (2) Launch the UI
import databricks_catalog_browser as dcb

dcb.launch_catalog_browser()  # optional: default_limit=50
```

Features
--------
* Navigate **catalog → schema → table** via dropdowns.
* Select **multiple tables** at once.
* **Ad‑hoc SQL textarea**: write any query.
* Adjustable **row‑limit control**.
* Results shown inline with `display()`.

Requirements: Databricks Runtime 14.x + and `ipywidgets` (pre‑installed on DBR 14 +). Works in classic and Jupyter‑compatible Databricks notebooks.
"""
from __future__ import annotations

import ipywidgets as widgets
from IPython.display import display, clear_output
from pyspark.sql import SparkSession

__all__ = ["launch_catalog_browser"]

# -----------------------------------------------------------------------------
# Spark session helper                                                          
# -----------------------------------------------------------------------------

def _spark() -> SparkSession:
    """Get (or create) the active SparkSession."""
    return SparkSession.builder.getOrCreate()


# -----------------------------------------------------------------------------
# Metadata fetchers                                                             
# -----------------------------------------------------------------------------

def _list_catalogs() -> list[str]:
    return [r.catalog_name for r in _spark().sql("SHOW CATALOGS").collect()]


def _list_schemas(catalog: str) -> list[str]:
    return [r.databaseName for r in _spark().sql(f"SHOW SCHEMAS IN {catalog}").collect()]


def _list_tables(catalog: str, schema: str) -> list[str]:
    rows = _spark().sql(f"SHOW TABLES IN {catalog}.{schema}").collect()
    return [r.tableName for r in rows]


# -----------------------------------------------------------------------------
# UI builder                                                                    
# -----------------------------------------------------------------------------

def _build_ui(default_limit: int = 20) -> widgets.VBox:
    """Create and wire up ipywidgets components."""

    # --- widgets ---
    catalog_dd = widgets.Dropdown(options=_list_catalogs(), description="Catalog:", layout=widgets.Layout(width="300px"))
    schema_dd = widgets.Dropdown(options=[], description="Schema:", layout=widgets.Layout(width="300px"))
    table_sel = widgets.SelectMultiple(options=[], description="Tables:", rows=8, layout=widgets.Layout(width="300px"))
    limit_int = widgets.BoundedIntText(value=default_limit, min=1, max=10000, description="Rows:", layout=widgets.Layout(width="200px"))
    run_btn = widgets.Button(description="Run Query", button_style="success")
    sql_box = widgets.Textarea(placeholder="Or write your own SQL here …", description="SQL:", layout=widgets.Layout(width="600px", height="120px"))
    out = widgets.Output()

    # --- callbacks ---
    def _on_catalog(change):
        catalog = change["new"]
        schema_dd.options = _list_schemas(catalog)
        if schema_dd.options:
            schema_dd.value = schema_dd.options[0]

    def _on_schema(change):
        catalog = catalog_dd.value
        schema = change["new"]
        table_sel.options = _list_tables(catalog, schema)

    def _run_query(_):
        with out:
            clear_output()
            sql = sql_box.value.strip()
            if not sql:
                if not table_sel.value:
                    print("⚠️ Select at least one table or type custom SQL.")
                    return
                refs = [f"{catalog_dd.value}.{schema_dd.value}.{t}" for t in table_sel.value]
                sql = "SELECT * FROM " + ", ".join(refs) + f" LIMIT {limit_int.value}"
            print("Executing:\n", sql)
            try:
                pdf = _spark().sql(sql).limit(limit_int.value).toPandas()
                display(pdf)
            except Exception as exc:
                print("Query failed:", exc)

    # wire observers
    catalog_dd.observe(_on_catalog, names="value")
    schema_dd.observe(_on_schema, names="value")
    run_btn.on_click(_run_query)

    # initial populate
    _on_catalog({"new": catalog_dd.value})

    # layout
    return widgets.VBox([
        widgets.HBox([catalog_dd, schema_dd]),
        widgets.HBox([table_sel, widgets.VBox([limit_int, run_btn])]),
        sql_box,
        out,
    ])


# -----------------------------------------------------------------------------
# Public API                                                                    
# -----------------------------------------------------------------------------

def launch_catalog_browser(default_limit: int = 20) -> None:
    """Display the catalog/table explorer in the current notebook cell."""
    display(_build_ui(default_limit))


# -----------------------------------------------------------------------------
# Script entry‑point                                                            
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    launch_catalog_browser()
