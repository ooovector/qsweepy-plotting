import random

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import timeit



start = timeit.timeit()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,
                # external_stylesheets=external_stylesheets,
                meta_tags= [{"name": "viewport",
                                    "content": "width=device-width, initial-scale=1"}])  # , static_folder='static')
app.config['suppress_callback_exceptions'] = True  # Set to `True` if your layout is dynamic, to bypass these checks.


@app.callback(
    Output(component_id='reload-page', component_property='children'),
    Input(component_id='update-button', component_property='n_clicks')
)
def update_measurements(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    else:
        measurements = get_measurements()
        # read input file
        fin = open("assets/scheme_source.svg", "r")
        # read file contents to string
        data = fin.read()
        # close the input file
        fin.close()

        for k, v in measurements.items():
            if k.split("_")[-1] == 'f':
                if k.split("_")[0] == "coupler":
                    data = data.replace(f'id="{k}">F = ???', f'id="{k}">F = {v}')
                elif k.split("_")[0] == "transmon":
                    data = data.replace(f'id="{k}">F   = ???', f'id="{k}">F   = {v}')
            elif k.split("_")[-1] == 't1':
                data = data.replace(f'id="{k}"> T1 = ???', f'id="{k}"> T1 = {v}')
            elif k.split("_")[-1] == 't2':
                data = data.replace(f'id="{k}"> T2 = ???', f'id="{k}"> T2 = {v}')

        # open the input file in write mode
        fin = open("assets/scheme_app.svg", "wt")
        # overrite the input file with the resulting data
        fin.write(data)
        # close the file
        fin.close()
        print('updated')

def get_measurements():
    return {
        'transmon_1_f': random.randint(0, 5),
        'transmon_1_t1': random.randint(0, 5),
        'transmon_1_t2': random.randint(0, 5),

        'transmon_2_f': random.randint(0, 5),
        'transmon_2_t1': random.randint(0, 5),
        'transmon_2_t2': random.randint(0, 5),

        'transmon_3_f': random.randint(0, 5),
        'transmon_3_t1': random.randint(0, 5),
        'transmon_3_t2': random.randint(0, 5),

        'transmon_4_f': random.randint(0, 5),
        'transmon_4_t1': random.randint(0, 5),
        'transmon_4_t2': random.randint(0, 5),

        'coupler_1_2_f': random.randint(0, 5),
        'coupler_2_3_f': random.randint(0, 5),
        'coupler_3_4_f': random.randint(0, 5),
        'coupler_4_1_f': random.randint(0, 5)
    }


def layout():
    return html.Div(children=[
            html.A(
                html.Button("UPDATE",
                        id='update-button',
                        n_clicks=0,
                        style={
                            'font-size': 18,
                            'color': 'RGB(71, 71, 71)',
                            'background-color': 'RGB(108, 172, 228)',
                            'border-radius': '6px',
                            'border': None,
                            'text-align': 'center',
                            'width': '100px',
                            'padding': '0px',
                            'margin': '0px'
                             }
                ),
                id='reload-page',
                href='/'),
            html.H1("4 QUBIT CHIP SUMMARY",
                    style={'font-size': 40,
                           'color': 'RGB(71, 71, 71)',
                           'text-align': 'center',
                           'padding': '0px',
                           'margin': '0px'
                           }
                    ),
            html.Img(id='scheme', src="/assets/scheme_app.svg", width="100%", height="100%"),
            html.Meta(httpEquiv="refresh", content="600"), #This command tells the html to refresh the page after n seconds.
        ],
            style={
                'position': 'fixed',
                'align': 'center',
                'width': '100%',
                'height': '100%',
                'background-color': 'RGB(108, 172, 228)',
                'padding': '0px',
                'margin': '0px'
        }
    )

end = timeit.timeit()
print('elapsed time', end - start)

if __name__ == '__main__':
    app.layout = layout()
    app.run_server(debug=False)
