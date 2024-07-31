from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import pandas as pd
import io
import base64
import datetime
from ratios import perform_financial_analysis  # Importing the analysis function from your ratios module

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# Global dictionary to store dataframes for analysis
uploaded_files = {'balance_sheet': None, 'income_statement': None, 'price_history': None}

app.layout = html.Div([
    html.H1("FINANCIAL RATIO CALCULATOR", style={'textAlign': 'center', 'marginTop': '20px'}),  # Added title with center alignment and margin
    html.H1("Upload Financial Documents", style={'marginTop': '15px'}),
    
    dcc.Upload(
        id='upload-balance-sheet',
        children=html.Div(['Drag and Drop or ', html.A('Select Balance Sheet')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False  # Single file upload
    ),
    html.Div(id='output-balance-sheet'),
    
    dcc.Upload(
        id='upload-income-statement',
        children=html.Div(['Drag and Drop or ', html.A('Select Income Statement')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False  # Single file upload
    ),
    html.Div(id='output-income-statement'),
    
    dcc.Upload(
        id='upload-price-history',
        children=html.Div(['Drag and Drop or ', html.A('Select Price History')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False  # Single file upload
    ),
    html.Div(id='output-price-history'),

    html.Button('ANALYZE', id='analyze-button', n_clicks=0, style={'display': 'none'}),
    html.Div(id='analysis-output')
])


@callback(
    Output('analyze-button', 'style'),
    [Input('output-balance-sheet', 'children'),
     Input('output-income-statement', 'children'),
     Input('output-price-history', 'children')]
)
def update_button_visibility(balance_sheet_output, income_statement_output, price_history_output):
    # Check if all outputs have content, which indicates files have been successfully uploaded
    if balance_sheet_output and income_statement_output and price_history_output:
        return {'display': 'block'}  # Show button
    return {'display': 'none'}  # Hide button

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'xlsx' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
            # Store DataFrame in global dictionary
            return html.Div([
                html.H5(filename),
                html.H6(datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')),
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df.columns],
                    style_table={'overflowX': 'auto'},
                    page_size=10
                ),
                html.Hr(),
            ])
    except Exception as e:
        return html.Div([
            f'There was an error processing this file: {str(e)}'
        ])

# Callbacks for uploading files
@callback(Output('output-balance-sheet', 'children'),
          Input('upload-balance-sheet', 'contents'),
          State('upload-balance-sheet', 'filename'),
          State('upload-balance-sheet', 'last_modified'))
def update_output_balance_sheet(contents, filename, date):
    if contents:
        uploaded_files['balance_sheet'] = pd.read_excel(io.BytesIO(base64.b64decode(contents.split(',')[1])))
        return parse_contents(contents, filename, date)
    return html.Div("Please upload a file.")

@callback(Output('output-income-statement', 'children'),
          Input('upload-income-statement', 'contents'),
          State('upload-income-statement', 'filename'),
          State('upload-income-statement', 'last_modified'))
def update_output_income_statement(contents, filename, date):
    if contents:
        uploaded_files['income_statement'] = pd.read_excel(io.BytesIO(base64.b64decode(contents.split(',')[1])))
        return parse_contents(contents, filename, date)
    return html.Div("Please upload a file.")

@callback(Output('output-price-history', 'children'),
          Input('upload-price-history', 'contents'),
          State('upload-price-history', 'filename'),
          State('upload-price-history', 'last_modified'))
def update_output_price_history(contents, filename, date):
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'xlsx' in filename:
                # Skip the first 15 rows
                df = pd.read_excel(io.BytesIO(decoded), skiprows=15)
                uploaded_files['price_history'] = df  # Storing it in a global dict
                return parse_contents(contents, filename, date)
        except Exception as e:
            return html.Div([
                'There was an error processing this file: {}'.format(e)
            ])
    return html.Div("Please upload a file.")




@callback(Output('analysis-output', 'children'),
          Input('analyze-button', 'n_clicks'),
          prevent_initial_call=True)
def perform_analysis(n_clicks):
    if not all(df is not None for df in uploaded_files.values()):
        return html.Div("Please upload all required files before analyzing.")

    try:
        df_balance_sheet = uploaded_files['balance_sheet']
        df_income_statement = uploaded_files['income_statement']
        df_price_history = uploaded_files['price_history']
        df_income_statement = unit_conversion(df_balance_sheet, df_income_statement)


        # Using the renamed function
        analysis_result = perform_financial_analysis(df_balance_sheet, df_income_statement, df_price_history)

        return dash_table.DataTable(
            data=analysis_result.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in analysis_result.columns],
            style_table={
                'overflowX': 'auto',
                'overflowY': 'auto',
                'height': 'auto',  # Adjust height based on the content
                'minWidth': '100%',  # Ensures the table uses all available space
            },
            style_cell={
                'textAlign': 'left',
                'minWidth': '150px',  # Minimum width for all columns
                'width': '150px',    # Width for all columns
                'maxWidth': '300px',  # Maximum width for all columns
            },
            style_cell_conditional=[
                {'if': {'column_id': 'Focus'}, 'width': '20%'},  # Adjust width for specific columns
                {'if': {'column_id': 'Metric'}, 'width': '50%'},
                {'if': {'column_id': 'Value'}, 'width': '30%'},
            ],
            page_size=len(analysis_result),  # Set page size to the length of the dataset to avoid pagination
            style_as_list_view=True,  # Optional: makes the table look more compact
        )
    except Exception as e:
        return html.Div(f"Error in analysis: {str(e)}")

# Ensure your app layout and other components are set up here

if __name__ == '__main__':
    app.run_server(debug=True)
