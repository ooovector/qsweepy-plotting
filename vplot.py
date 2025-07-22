import qsweepy
from qsweepy import *
from qsweepy.ponyfiles import *
import dash
from conf import *
from dash import dcc
from dash import html
from dash import dash_table
import numpy as np
import webcolors
import datetime
from qsweepy.libraries.plotly_plot import *

import exdir
from qsweepy.ponyfiles.data_structures import *
import plotly.graph_objs as go
from pony.orm import *
import plotly.io as pio
from qsweepy.ponyfiles import database
from plotly import *
from cmath import phase
from dash.dependencies import Input, Output, State
import pandas as pd
import psycopg2
from psycopg2 import sql
import pandas.io.sql as psql
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

import sys
sys.path.append("C:\\qtlab_replacement\\qsweepy\\qsweepy\\libraries")
from plotly_plot import plot as reduced_plot

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)  # , static_folder='static')
app.config['suppress_callback_exceptions'] = True  # Set to `True` if your layout is dynamic, to bypass these checks.
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
db = database.MyDatabase()

def measurement_table():
    return dash_table.DataTable(
        id="meas-id",
        columns=[{'id':'id', 'name':'id'}, {'id':'label', 'name':'label'}],
        data=default_measurements(db),
        editable=True,
        row_deletable=True,
        row_selectable='single')


@app.callback(
    Output(component_id="meas-id", component_property="data"),
    Input(component_id="query-results-table", component_property="derived_virtual_data"),
    Input(component_id="query-results-table", component_property="derived_virtual_selected_rows"),
    State(component_id="meas-id", component_property="derived_virtual_data")
)
def render_measurement_table(query_results, query_results_selected, current_measurements):
    # print ('render_measurement_table called')
    # current_measurements = []
    if current_measurements is None:
        return []

    # print(query_results, query_results_selected, current_measurements)
    if query_results is None:
        query_results = []
    if query_results_selected is None:
        query_results_selected = []
    selected_measurement_ids = [query_results[measurement]['id'] for measurement in query_results_selected]
    deselected_measurement_ids = [measurement['id'] for measurement in query_results if
                                  not measurement['id'] in selected_measurement_ids]
    old_measurement_ids = [measurement['id'] for measurement in current_measurements]
    old_measurements = [measurement for measurement in current_measurements if
                        not measurement['id'] in deselected_measurement_ids]
    new_measurements = [{'id': query_results[measurement]['id'],
                         'label': (query_results[measurement]['label'] if 'label' in query_results[measurement] else
                                   query_results[measurement]['id'])}
                        for measurement in query_results_selected if
                        (not query_results[measurement]['id'] in old_measurement_ids)]

    return old_measurements + new_measurements


def available_traces_table(data=[], column_static_dropdown=[], column_conditional_dropdowns=[], selected_rows=None, style_conditions=[]):
    if selected_rows is None:
        selected_rows = np.arange(len(data))
    return dash_table.DataTable(id="available-traces-table",
                                data=data,
                                columns=[
                                    {'id': 'id', 'name': 'id'},
                                    {'id': 'dataset', 'name': 'dataset'},
                                    {'id': 'op', 'name': 'op'},
                                    {'id': 'style', 'name': 'style', 'presentation': 'dropdown'},
                                    {'id': 'color', 'name': 'color', 'presentation': 'dropdown'},
                                    {'id': 'x-axis', 'name': 'x-axis', 'presentation': 'dropdown'},
                                    {'id': 'y-axis', 'name': 'y-axis', 'presentation': 'dropdown'},
                                    {'id': 'row', 'name': 'row'},
                                    {'id': 'col', 'name': 'col'}],
                                editable=True,
                                row_selectable='multi',
                                selected_rows=selected_rows,
                                dropdown=column_static_dropdown,
                                dropdown_conditional=column_conditional_dropdowns,
                                style_data_conditional=style_conditions)


@app.callback(
    Output(component_id="available-traces-container", component_property="children"),
    Input(component_id="meas-id", component_property="derived_virtual_data"),
     Input(component_id="update-available-traces", component_property="n_clicks"),
    State(component_id="available-traces-table", component_property="derived_virtual_data"),
   State(component_id="available-traces-table", component_property="data"),
   State(component_id="available-traces-table", component_property="derived_virtual_selected_rows"),
   State(component_id="available-traces-table", component_property="selected_rows"),
   State(component_id="available-traces-table", component_property="dropdown_conditional")
)
def render_available_traces_table(loaded_measurements, intermediate_value_meas, current_traces_modified, current_traces,
                                  current_selected_traces_modified, current_selected_traces,
                                  current_conditional_dropdowns):
    print("LOL NCLICKS UPDATE TRACES")
    print("current_selected_traces_modified", current_selected_traces_modified, "current_selected_traces", current_selected_traces)
    # if old state exists, start with it, otherwise default to empty selection
    if loaded_measurements is None: loaded_measurements = []
    if current_traces_modified: current_traces = current_traces_modified
    if current_selected_traces_modified: current_selected_traces = current_selected_traces_modified

    if len(current_selected_traces):
        old_traces = [current_traces[i] for i in current_selected_traces]
    else:
        old_traces = []

    if current_conditional_dropdowns is None:
        current_conditional_dropdowns = []
    elif len(current_conditional_dropdowns):
        conditional_dropdowns = current_conditional_dropdowns  # [0]['dropdowns']
    else:
        conditional_dropdowns = []

    # colors = [c for c in webcolors.CSS3_NAMES_TO_HEX.keys()]

    colors = ['black',
              'red',
              'blue',
              'green',
              'yellow',
              'maroon',
              'fuchsia',
              'deeppink',
              'lime',
              'navy',
              'teal',
              'aqua',
              'coral',
              'cornflowerblue',
              'gold',
              'lightseagreen',
              'orangered',
              'slateblue',
              'yellowgreen']

    styles = ['2d', '-', '.', 'o']

    data, conditional_dropdowns = add_default_traces(loaded_measurements=loaded_measurements,
                                                     db=db, old_traces=old_traces,
                                                     conditional_dropdowns=conditional_dropdowns)
    column_static_dropdown = {
        'style': {'options': [{'label': s, 'value': s} for s in styles]},
        'color': {'options': [{'label': c, 'value': c} for c in colors]}}

    style_conditions = [
        {
            'if': {'column_id': 'color',
                   'filter_query': '{color} = ' + c},
            'backgroundColor': c
        } for c in colors]

    selected_rows = np.arange(len(old_traces)) if len(current_traces) else np.arange(len(data))

    return available_traces_table(data=data,
                                  column_static_dropdown=column_static_dropdown,
                                  column_conditional_dropdowns=conditional_dropdowns,
                                  # [{'id': 'x-axis', 'dropdowns': conditional_dropdowns},
                                  #  {'id': 'y-axis', 'dropdowns': conditional_dropdowns}],
                                  selected_rows=selected_rows,
                                  style_conditions=style_conditions)

@app.callback(Output('cross-section-configuration', 'data'),
              Input(component_id="available-traces-table", component_property="derived_virtual_data"),
               Input(component_id="available-traces-table", component_property="data"),
               Input(component_id="available-traces-table", component_property="derived_virtual_selected_rows"),
               Input(component_id="available-traces-table", component_property="selected_rows"),
              State(component_id='cross-section-configuration', component_property='derived_virtual_data'))
def cross_section_configuration_data(all_traces, all_traces_initial, selected_trace_ids, selected_trace_ids_initial,
                                     current_config):
    if not all_traces:
        all_traces = all_traces_initial
    if not selected_trace_ids:
        selected_trace_ids = selected_trace_ids_initial
    selected_traces = pd.DataFrame([all_traces[i] for i in range(len(all_traces)) if i in selected_trace_ids],
                                   columns=['id', 'dataset', 'op', 'style', 'color', 'x-axis', 'y-axis', 'row', 'col'])
    return cross_section_configurations_add_default(selected_traces, db, current_config)


def cross_section_configuration(data=[]):
    return dash_table.DataTable(id='cross-section-configuration',
                                columns=[{'id': 'trace-id', 'name': 'trace #'},
                                         {'id': 'parameter-id', 'name': 'parameter #'},
                                         {'id': 'parameter', 'name': 'parameter'},
                                         {'id': 'value', 'name': 'value'}],
                                data=data,
                                editable=True)


def app_layout():
    return html.Div(children=[
        dcc.Location(id='url', refresh=False),
        html.Div(id="modal-select-measurements", className="modal", style={'display': 'none'},
                 children=modal_content()),
        html.Div([
            # html.H1(id = 'list_of_meas_types', style = {'fontSize': '30', 'text-align': 'left', 'text-indent': '5em'}),
            dcc.Graph(id='live-plot-these-measurements', style={'height': '100%', 'width': '70%'})],
            style={'position': 'absolute', 'width': '98%', 'height': '98%'}),
        # style = {'position': 'absolute', 'top': '30', 'left': '30', 'width': '1500' , 'height': '1200'}),
        html.Div([
            html.Div([html.H4('Measurements: '), measurement_table()]),
            html.Button(id='modal-select-measurements-open', children=['Add measurements...'], n_clicks=0),
            html.Button(id='update-available-traces', children=['Update available traces']),
            html.Button(id='deselect-all-button', children=['Deselect all traces']),
            html.Button(id='save-svg', children=['Save svg to C:']),
            html.Div(id='hidden-div-save-svg', style={'display': 'none'}),
            html.Data(id='counter-deselect-all-clicks', value=0, style={'display': 'none'}),
            html.Div(id='table_of_meas'),
            html.H4(children='Measurement info'),
            html.Div(id='meas_info'),
            html.Div(children=[html.H4('Available traces: '),
                               html.Div(id='available-traces-container', children=[available_traces_table()])]),
            # dcc.Input(id='meas-id2', value = str(meas_ids), type = 'string')]),
            # html.Div([html.P('You chose following fits: '), dcc.Input(id='fit-id', value = str(fit_ids), type = 'string')]),
            html.Div(id='cross-section-configuration-container', children=[cross_section_configuration()])
        ],
            style={'position': 'absolute', 'top': '5%', 'left': '68%', 'width': '30%', 'height': '88%',
                   # 'position': 'absolute', 'top': '80', 'left': '1500', 'width': '350' , 'height': '800',
                   'padding': '0px 10px 15px 10px',
                   'marginLeft': 'auto', 'marginRight': 'auto',  # 'background': 'rgba(167, 232, 170, 1)',
                   'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)'},  # rgba(190, 230, 192, 1)'},
        ),
        dcc.Interval(
            id='interval-component',
            interval=3 * 1000,  # in milliseconds
            n_intervals=0
        ),
        html.Div(id='intermediate-value-meas'),  # , style={'display': 'none'}),
    ])


@app.callback(
    Output(component_id='hidden-div-save-svg', component_property="children"),
    Input(component_id="save-svg", component_property="n_clicks"),
    State('live-plot-these-measurements', 'figure')
)
def save_svg(n_clicks, figure):
    import uuid
    if n_clicks is None:
        return []
    unique_filename = str(uuid.uuid4())
    pio.write_image(figure, PATH + "{}.svg".format(unique_filename), width=1200, height=900)
    return []


@app.callback(
    Output("available-traces-table", "selected_rows"),
    Input('deselect-all-button', 'n_clicks'),
    State(component_id='counter-deselect-all-clicks', component_property='value')
)
def deselect_all(n_clicks, n_clicks_saved):
    print("LOL NCLICKS", n_clicks, n_clicks_saved)
    if n_clicks == n_clicks_saved:
        raise dash.exceptions.PreventUpdate
    else:
        return []

@app.callback(
    Output(component_id='counter-deselect-all-clicks', component_property='value'),
    Input(component_id='deselect-all-button', component_property='n_clicks'))
def save_del_click_counter(n_clicks_deselect_all):
    return n_clicks_deselect_all

@app.callback(
    Output(component_id='meas_info', component_property='children'),
    [Input(component_id='meas-id', component_property='derived_virtual_data'),
     Input(component_id='meas-id',
           component_property="derived_virtual_selected_rows")])  # Input(component_id = 'my_dropdown', component_property='value')])
def write_meas_info(measurements, selected_measurement):
    try:
        value = measurements[selected_measurement[0]]['id']
    except:
        return []
    with db_session:
        if value == None: return
        state = save_exdir.load_exdir(db.Data[int(value)].filename, db)

        references = pd.DataFrame([{'this': i.this.id, 'that': i.that.id, 'ref_type': i.ref_type}
                                   for i in select(ref for ref in db.Reference if ref.this.id == int(value))],
                                  columns=['this', 'that', 'ref_type'])
        metadata = pd.DataFrame([(k, v) for k, v in state.metadata.items()], columns=['name', 'value'],
                                index=np.arange(len(state.metadata)), dtype=object)

        # print(state.metadata)

        # print (metadata.to_dict('records'), state.metadata)
        retval = [html.P(['Start: ' + state.start.strftime('%d-%m-%Y %H:%M:%S.%f'),
                          html.Br(),
                          'Stop: ' + state.stop.strftime('%d-%m-%Y %H:%M:%S.%f')]),
                  # html.Div(html.P('Owner: ' + str(state.owner))),
                  html.Div(children=[
                      html.H6('Metadata'),
                      dash_table.DataTable(
                          columns=[{"name": col, "id": col} for col in metadata.columns],
                          data=metadata.to_dict('records'),  ### TODO: add selected rows from meas-id row
                          id="metadata"
                      )],
                      id='metadata-container'
                  ),
                  html.Div(children=[
                      html.H6('References'),
                      dash_table.DataTable(
                          columns=[{"name": col, "id": col} for col in references.columns],
                          data=references.to_dict('records'),  ### TODO: add selected rows from meas-id row
                          id="references"
                      )],
                      id='references-container'
                  )
                  ]
        print(retval, metadata.to_dict('records'), references.to_dict('records'))
        return retval


@app.callback(Output('live-plot-these-measurements', 'figure'),
                Input(component_id='cross-section-configuration', component_property='derived_virtual_data'),
                Input(component_id="available-traces-table", component_property="derived_virtual_data"),
                # State(component_id="available-traces-table", component_property="derived_virtual_data"),
                State(component_id="available-traces-table", component_property="data"),
                State(component_id="available-traces-table", component_property="derived_virtual_selected_rows"),
                State(component_id="available-traces-table", component_property="selected_rows"),

                  # Input('interval-component', 'n_intervals')
              )
def render_plots(cross_sections, all_traces, all_traces_initial, selected_trace_ids, selected_trace_ids_initial,
                 # , n_intervals
                 ):
    from time import time
    start_time = time()
    if not all_traces:
        all_traces = all_traces_initial
    if not selected_trace_ids:
        selected_trace_ids = selected_trace_ids_initial
    # print ('all_traces: ', all_traces)
    # print ('all_traces_initial: ', all_traces_initial)
    # print ('selected_trace_ids: ', selected_trace_ids)
    # print ('cross_sections: ', cross_sections)
    selected_traces = pd.DataFrame([all_traces[i] for i in range(len(all_traces)) if i in selected_trace_ids],
                                   columns=['id', 'dataset', 'op', 'style', 'color', 'x-axis', 'y-axis', 'row', 'col'])
    p = reduced_plot(selected_traces, cross_sections, db, max_data_size=1.5e6)
    # p = plot(selected_traces, cross_sections, db)
    end_time = time()
    print('render_plots time: ', end_time - start_time)
    return p


def get_queries(columns=None):
    try:
        direct_db = psycopg2.connect(database='qsweepy',
                                     user='qsweepy',
                                     password='qsweepy')
        saved_queries = psql.read_sql(EXTRACT_QUERIES, direct_db)
        if columns:
            return saved_queries[columns]
        else:
            return saved_queries

    except Exception as e:
        error = str(e)
        return html.Div(children=error)
    finally:
        direct_db.close()


def modal_content():
    return [html.Div(className="modal-content",
                     children=[
                         html.Div(className="modal-header", children=[
                             html.Span(className="close", children="×", id="modal-select-measurements-close", n_clicks=0),
                             html.H1(children="Measurements query"),
                         ]),
                         html.Div(className="modal-body", children=[
                             html.Div(className="modal-left", children=[
                                 html.Div("Saved queries"),
                                 dcc.RadioItems(
                                     id='query-names-list',
                                     options=[{'label': n, 'value': n} for n in get_queries('query_name')
                                              ],
                                     value=None
                                 ),
                             ]),
                             html.Div(className="modal-right", children=[
                                 html.Div(className="modal-right-content", children=[
                                     html.Div(
                                         children=[dcc.Textarea(id='query', value=DEFAULT_QUERY, style={'width': '100%', 'height': 100})]),
                                     html.Div(children=[html.Button('Execute', id='execute'),
                                                        html.Button('Deselect all', id='deselect-all-button-meas-table', n_clicks=0),
                                                        html.Data(id='counter-deselect-all-meas-query-table-clicks', value=dict(
                                                            {'n_clicks_select_all': 0, 'n_clicks_deselect_all': 0}),
                                                                  style={'display': 'none'})
                                                        ]),
                                     html.Div(["Query name: ",
                                               dcc.Input(id='query-name', type='text'),
                                               html.Button('Save query', id='save-query', type='submit', n_clicks=0),
                                               html.Data(id='counter-save-del-clicks', value=dict(
                                                   {'n_clicks_save_query': 0, 'n_clicks_delete_query': 0}),
                                                         style={'display': 'none'}),
                                               html.Button('Delete query', id='delete-query', type='submit',
                                                           n_clicks=0),
                                               ]),
                                     html.Div(id='query-results', className='query-results', children=[]),
                                 ]),
                             ])
                         ]),
                         # html.Div(className="modal-footer", children=["Modal footer"])
                     ])]


# n_clicks_registered = 0

@app.callback(
    Output(component_id='query-name', component_property='value'),
    [Input(component_id='query-names-list', component_property='value')])
def auto_fill_query_name(query_name):
    return query_name


@app.callback(
    Output(component_id='query', component_property='value'),
    [Input(component_id='query-names-list', component_property='value')])
def update_query(query_name):
    saved_queries = get_queries()
    return saved_queries[saved_queries['query_name'] == query_name]['query'].iloc[0] if query_name in list(saved_queries['query_name']) else DEFAULT_QUERY


@app.callback(
    Output(component_id='query-results', component_property='children'),
    Input(component_id='execute', component_property='n_clicks'),
    Input(component_id='modal-select-measurements-open', component_property='n_clicks'),
    Input(component_id='deselect-all-button-meas-table', component_property='n_clicks'),
    State(component_id='query', component_property='value'),
    State(component_id='counter-deselect-all-meas-query-table-clicks', component_property='value'),
    State(component_id='meas-id', component_property='derived_virtual_data')
)
def update_query_result(n_clicks_execute, n_clicks_select_measurements_open, n_clicks_deselect_all, query, n_clicks_deselect_all_counter, selected_measurements):
    # global n_clicks_registered
    # if n_clicks> n_clicks_registered:
    # n_clicks_registered = n_clicks
    selected_measurements = pd.DataFrame(selected_measurements, columns=['id', 'label'])
    try:
        direct_db = psycopg2.connect(database='qsweepy', user='qsweepy', password='qsweepy')
        dataframe = psql.read_sql(query, direct_db)

        if n_clicks_deselect_all == n_clicks_deselect_all_counter:# and 'id' in dataframe.columns:
            old_measurements = [row_id for row_id, i in enumerate(dataframe['id'].tolist()) if
                                i in selected_measurements['id'].tolist()]
        else:
            old_measurements = []
        modal_content()
        return [html.Div(className="query-results-scroll",
                         children=[dash_table.DataTable(
                             # Header
                             columns=[{"name": col, "id": col} for col in dataframe.columns],
                             style_data_conditional=[{"if": {"column_id": 'id'},
                                                      'background-color': '#c0c0c0',
                                                      'color': 'white'}],
                             data=dataframe.to_dict('records'),
                             row_selectable='multi',
                             selected_rows=old_measurements,  ### TODO: add selected rows from meas-id row
                             id="query-results-table"
                         )]
                         )]
    except Exception as e:
        error = str(e)
        return html.Div(children=error)
    finally:
        direct_db.close()


@app.callback(
    Output(component_id='counter-deselect-all-meas-query-table-clicks', component_property='value'),
    Input(component_id='deselect-all-button-meas-table', component_property='n_clicks'))
def select_deselect_all_click_counter(n_clicks_deselect_all):
    return n_clicks_deselect_all

@app.callback(
    Output(component_id='counter-save-del-clicks', component_property='value'),
    [Input(component_id='save-query', component_property='n_clicks'),
     Input(component_id='delete-query', component_property='n_clicks')])
def save_del_click_counter(n_clicks_save_query, n_clicks_delete_query):
    return {'n_clicks_save_query': n_clicks_save_query,
            'n_clicks_delete_query': n_clicks_delete_query}


@app.callback(
    Output(component_id='query-names-list', component_property='options'),
    Input(component_id='save-query', component_property='n_clicks'),
    Input(component_id='delete-query', component_property='n_clicks'),
    State(component_id='query', component_property='value'),
    State(component_id='query-name', component_property='value'),
    State(component_id='counter-save-del-clicks', component_property='value')
)
def save_delete_query(n_clicks_save_query, n_clicks_delete_query, query, query_name, saved_n_clicks):
    query_date = datetime.now(tz=None).strftime("%Y-%m-%d %H:%M:%S")
    try:
        direct_db = psycopg2.connect(database='qsweepy',
                                     user='qsweepy',
                                     password='qsweepy')
        cur = direct_db.cursor()
        saved_queries = psql.read_sql(EXTRACT_QUERIES, direct_db)
        if n_clicks_save_query > saved_n_clicks['n_clicks_save_query']:
            if not query_name:
                query_name = DEFAULT_QUERY_NAME_PREFIX + query_date
            saved_queries = saved_queries.append({'query_name': query_name, 'query': query, 'query_date': query_date},
                                                 ignore_index=True)
            cur.execute("""INSERT INTO queries (query_name, query, query_date) VALUES (%s, %s, %s);""",
                        (query_name, query, query_date))
            print(f"{query_name} has been saved")

        elif n_clicks_delete_query > saved_n_clicks['n_clicks_delete_query']:
            saved_queries = saved_queries[saved_queries['query_name'] != query_name]
            cur.execute("""DELETE FROM queries WHERE query_name = %s;""", (query_name,))
            print(query_name, 'has been deleted')

        return [{'label': n, 'value': n} for n in saved_queries['query_name']]

    except Exception as e:
        error = str(e)
        return html.Div(children=error)
    finally:
        direct_db.commit()
        cur.close()
        direct_db.close()


@app.callback(
    Output(component_id='modal-select-measurements', component_property='style'),
    [Input(component_id='modal-select-measurements-open', component_property='n_clicks'),
     Input(component_id='modal-select-measurements-close', component_property='n_clicks')]
)
def modal_select_measurements_open_close(n_clicks_open, n_clicks_close):
    return {'display': 'block' if (n_clicks_open - n_clicks_close) % 2 else 'none'};


if __name__ == '__main__':
    app.layout = app_layout()
    app.run(host='0.0.0.0', debug=False, port=8060)
    # app.run_server(debug=False)
