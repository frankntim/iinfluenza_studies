dbc.Container([
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardImg(src="/assets/general_analysis.png", top=True, style={"height": "180px", "width": "100%", "objectFit": "cover"}),
                dbc.CardBody(html.H5("General Analysis", className="card-title")),
                dbc.CardFooter(dbc.Button("Run Analysis", color="primary", id="run-general"))
            ]),
            width=6
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardImg(src="/assets/survival_analysis.png", top=True, style={"height": "180px", "width": "100%", "objectFit": "cover"}),
                dbc.CardBody(html.H5("Survival Analysis", className="card-title")),
                dbc.CardFooter(dbc.Button("Run Analysis", color="primary", id="run-survival"))
            ]),
            width=6
        )
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardImg(src="/assets/cox_fitting.png", top=True, style={"height": "180px", "width": "100%", "objectFit": "cover"}),
                dbc.CardBody(html.H5("Cox Fitting Analysis", className="card-title")),
                dbc.CardFooter(dbc.Button("Run Analysis", color="primary", id="run-cox"))
            ]),
            width=6
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardImg(src="/assets/linear_mixed.png", top=True, style={"height": "180px", "width": "100%", "objectFit": "cover"}),
                dbc.CardBody(html.H5("Linear Mixed Analysis", className="card-title")),
                dbc.CardFooter(dbc.Button("Run Analysis", color="primary", id="run-lmm"))
            ]),
            width=6
        )
    ])
])
