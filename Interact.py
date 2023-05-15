# MIT License
#
# Copyright (c) 2023 Nicolette Shaw
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import base64
import io

from ncempy.io import dm
import numpy as np
import matplotlib.pyplot as plt

from bokeh.io import curdoc, output_file, show
from bokeh.layouts import column, row
from bokeh.models.widgets import Button, FileInput, Toggle, RadioButtonGroup
from bokeh.models import ColumnDataSource, Slider, TextInput, Range1d, CustomJS
from bokeh.plotting import figure, gridplot, output_file, show
from tornado.ioloop import IOLoop
from bokeh.models.ranges import FactorRange

from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.layouts import column
from bokeh.server.server import Server
from scipy.integrate import simps


# ds = dm.dmReader("data/EELS Spectrum Image.dm4")
# plt.plot(ds['data'][0])
# plt.show()

# ds['coords'], [0] = pixels, [1] = energy of spectra (['pixelSize'][1] step).

def alignZLPbyHighestPeak(ds, index_range: list):
    # x = ds['xdata'][0]
    # y = ds['data'][0]

    x_step = ds['pixelSize'][1]
    n_data_points = len(ds['xdata'])
    for i in range(0, len(ds['data'])):
        # Get index for peak in range in y data.
        arr = ds['data'][i][index_range[0], index_range[1]]
        peak_index = arr.index(max(arr)) + index_range[0]

        # Shift x data so peak appears at x = 0.
        if ds['xdata'][peak_index] == 0:
            continue
        else:
            starting_x = - peak_index * x_step
            ds['xdata'] = range(starting_x, x_step * n_data_points, x_step)

    return ds


def TotalAreaAcrossScan(ds):
    areas = []
    for i in range(0, len(ds['data'])):
        areas.append(simps(ds['data'][i]))
    return ds['coords'][0], areas


ds = {}
tools = "pan,wheel_zoom,reset,hover,xbox_select,lasso_select"


def serverFunction(doc):
    source = ColumnDataSource(data=dict(x=[], y=[]))
    # Set up plot
    plot = figure(height=400, width=400, title="my sine wave",
                  tools=tools,
                  x_range=[0, 2000], y_range=[-50, 5000])

    plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

    spectrum_num = Slider(title="Scan point", value=1, start=1, end=10, step=1)

    totalArea = Toggle(label="Show total plot area", button_type="success")

    range_start = TextInput(value="", title="Range start:")
    range_end = TextInput(value="", title="Range end:")
    alignZLPbutton = Button(label="Align ZLP", button_type="danger")

    ## Set up callbacks

    # Files must be local for upload to work.
    def upload_file_func(attrname, old, new):

        global ds

        # Decode file info.
        decode = base64.b64decode(new)
        df = io.BytesIO(decode)
        # Keep temp file until app closes.
        with open('file.temp.dm4', 'wb') as f:
            f.write(df.getvalue())
        ds = dm.dmReader('file.temp.dm4')

        # Create separate x axes for each point in linescan under ds['xdata'].
        ds['xdata'] = []
        for i in range(0, len(ds['data'])):
            ds['xdata'].append(ds['coords'][1])

        # Access the first plot in a linescan.
        x = ds['xdata'][0]
        y = ds['data'][0]

        # Update axis scales.
        plot.x_range.start = min(x)
        plot.x_range.end = max(x)
        plot.y_range.start = min(y)
        plot.y_range.end = max(y)

        # Refresh plot data.
        source.data = dict(x=x, y=y)

        # Reset scan point range.
        spectrum_num.end = len(ds['data'])

    # Scan across linescan to different points.
    def update_point(attrname, old, new):
        if not ds:
            return

        # Get new point value in linescan.
        point = spectrum_num.value

        # Refresh to new spectrum.
        x = ds['xdata'][point]
        y = ds['data'][point]

        # Refresh plot data.
        source.data = dict(x=x, y=y)

    def show_totalAreaPlot(active):
        if not ds:
            return

        if active:
            x_totalArea, y_totalArea = TotalAreaAcrossScan(ds)

            # Set up plot
            areasource = ColumnDataSource(data=dict(x=x_totalArea, y=y_totalArea))
            area_plot = figure(height=400, width=400, title="my sine wave",
                               tools=tools,
                               x_range=[min(x_totalArea), max(x_totalArea)],
                               y_range=[min(y_totalArea), max(y_totalArea)])

            area_plot.line('x', 'y', source=areasource, line_width=3)
            show(area_plot)

    def highlightRange(attrname, old, new):
        print('yes')
        start = range_start.value
        end = range_end.value
        point = spectrum_num.value

        # TODO not working

    def alignZLPinRange(active):  # TODO not working

        global ds

        if not ds:
            return
        if not range_start.value:
            return
        if not range_end.value:
            return

        # Get new point value in linescan.
        point = spectrum_num.value

        start = range_start.value
        end = range_end.value

        ds = alignZLPbyHighestPeak(ds, [start, end])

        # Refresh to new spectrum.
        x = ds['xdata'][point]
        y = ds['data'][point]

        # Refresh plot data.
        source.data = dict(x=x, y=y)

    # Attatch file when chosen.
    file_input = FileInput()
    file_input.on_change('value', upload_file_func)

    # Move along linescan to different spectra.
    spectrum_num.on_change('value', update_point)

    # Toggle to show total area plot.
    totalArea.on_click(show_totalAreaPlot)

    # Highlight area of plot in text ranges.
    range_start.on_change('value', highlightRange)
    range_end.on_change('value', highlightRange)

    # Align ZLP.
    alignZLPbutton.on_click(alignZLPinRange)

    # Set up layouts and add to document.
    inputs = column(file_input, spectrum_num, totalArea, row(range_start, range_end, alignZLPbutton))
    doc.add_root(row(inputs, plot, width=800))
    doc.title = "Process EELS linescan data"


io_loop = IOLoop.current()

bokeh_app = Application(FunctionHandler(serverFunction))

server = Server({'/': bokeh_app}, io_loop=io_loop, websocket_max_message_size=50 * 1024 * 1024)
server.start()

if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')
    io_loop.add_callback(server.show, "/")
    io_loop.start()

#
# line_rend = plot.line('x', 'y', legend_label="Current", line_width=1, source=source)
# line_rend.selection_glyph = line_rend.glyph
# line_rend.nonselection_glyph = line_rend.glyph
#
# # now make a scatter renderer with zero alpha driving off the same ColumnDataSource
# scatter_rend = plot.scatter('x', 'y', fill_alpha=0, source=source, line_alpha=0)
# # do the the exact same thing as about with the selection glyphs and non selection glyphs
# scatter_rend.selection_glyph = scatter_rend.glyph
# scatter_rend.nonselection_glyph = scatter_rend.glyph
#
# # now create a "selection source" (you had something like this already)
# # initialize with no data
# sel_src = ColumnDataSource(data={'x': [], 'y': []})
# # make a renderer running off this source, orange line
# sel_line_render = plot.line('x', 'y', legend_label='Selected', line_color='orange', source=sel_src)
#
# # now the JS component
# # basically the alpha 0 scatter glyph will allow the selection tool to grab selected indices from s1
# # we use those selected indices to collect the corresponding values from s1 for the time and data fields
# # and push those values into arrays ("sel_x" and "sel_y")
# # use Math.min etc to get the min/max values from that array... (not sure what you want to do with it but I have it logging in the console)
# # then use the sel arrays to populate the sel_src, which your orange line is running off of... so it'll do what you want
# cb = CustomJS(args=dict(source=source, sel_src=sel_src)
#               , code='''
#             var sel_inds = source.selected.indices
#             var sel_x = []
#             var sel_y = []
#             for (var i=0;i<s1.selected.indices.length;i++){
#                     sel_x.push(source.data['x'][sel_inds[i]])
#                     sel_y.push(source.data['y'][sel_inds[i]])}
#             console.log('Min of selection:')
#             console.log(Math.min(...sel_data))
#             sel_src.data['x']= sel_x
#             sel_src.data['y'] = sel_y
#             sel_src.change.emit()
#             ''')
#
# # tell this callback to happen whenever the selected indices of s1 change
# source.selected.js_on_change('indices', cb)
# #
