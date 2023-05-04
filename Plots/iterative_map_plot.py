from PymoNNto.Exploration.Network_UI.TabBase import *

from pyqtgraph.Qt import QtCore, QtGui

class iterative_map_tab(TabBase):

    def __init__(self, x_var, y_var, x_time_shift=0, y_time_shift=0, title='ItMap', timesteps=500):
        super().__init__(title)
        self.x_var = x_var#self.interpret_recording_variable(x_var)
        self.y_var = y_var#self.interpret_recording_variable(y_var)
        self.x_time_shift=x_time_shift
        self.y_time_shift=y_time_shift
        self.timesteps=timesteps

    def add_recorder_variables(self, neuron_group, Network_UI):
        Network_UI.add_recording_variable(neuron_group, self.x_var, timesteps=self.timesteps+1)
        Network_UI.add_recording_variable(neuron_group, self.y_var, timesteps=self.timesteps+1)

    def initialize(self, Network_UI):
        self.itm_tab = Network_UI.add_tab(self.title)

        self.plot = Network_UI.tab.add_plot(x_label=self.x_var, y_label=self.y_var)

        self.lines = pg.PlotCurveItem(pen=(100,100,100,255))
        self.plot.addItem(self.lines)
        self.scatter = pg.ScatterPlotItem()
        self.plot.addItem(self.scatter)

        Network_UI.tab.add_row()
        Network_UI.tab.add_widget(QLabel('steps: '))
        self.step_slider = QSlider(1)  # QtCore.Horizontal
        self.step_slider.setMinimum(1)
        self.step_slider.setMaximum(self.timesteps)
        self.step_slider.setSliderPosition(self.timesteps)
        self.step_slider.setToolTip('slide to change step count')
        Network_UI.tab.add_widget(self.step_slider)  # , stretch=0.1

        self.grad_cb = Network_UI.tab.add_widget(QCheckBox('gradient'))
        self.grad_cb.setChecked(True)

        self.scatter_cb = Network_UI.tab.add_widget(QCheckBox('scatter'))
        self.scatter_cb.setChecked(True)

        self.lines_cb = Network_UI.tab.add_widget(QCheckBox('lines'))
        self.lines_cb.setChecked(True)

        self.x_shift_cb = Network_UI.tab.add_widget(QCheckBox('x t-1'))

        self.y_shift_cb = Network_UI.tab.add_widget(QCheckBox('y t-1'))



        self.step_slider.mouseReleaseEvent = Network_UI.static_update_func
        self.grad_cb.stateChanged.connect(Network_UI.static_update_func)
        self.scatter_cb.stateChanged.connect(Network_UI.static_update_func)
        self.lines_cb.stateChanged.connect(Network_UI.static_update_func)
        self.x_shift_cb.stateChanged.connect(Network_UI.static_update_func)
        self.y_shift_cb.stateChanged.connect(Network_UI.static_update_func)



    def update(self, Network_UI):
        if self.itm_tab.isVisible():
            groups = Network_UI.get_visible_neuron_groups()

            self.lines.setData([], [])
            self.scatter.setData([], [])

            if len(groups)>=2:
                x_group = groups[0]#Network_UI.network[group_tags[1], 0]####################################################################add selector
                y_group = groups[1]#Network_UI.network[group_tags[2], 0]####################################################################add selector

                if x_group is not None and y_group is not None:

                    try:
                        x_values = x_group[self.x_var, 0, 'np'][-self.step_slider.value():]
                        y_values = y_group[self.y_var, 0, 'np'][-self.step_slider.value():]

                        x_val = np.mean(x_values, axis=1)
                        y_val = np.mean(y_values, axis=1)

                        tx='(t)'
                        ty='(t)'
                        if self.x_shift_cb.checkState():
                            x_val = x_val[:-1]
                            y_val = y_val[1:]
                            tx = '(t-1)'

                        if self.y_shift_cb.checkState():
                            x_val = x_val[1:]
                            y_val = y_val[:-1]
                            ty = '(t-1)'

                        self.plot.getAxis('bottom').setLabel(text=self.x_var+tx)
                        self.plot.getAxis('left').setLabel(text=self.y_var+ty)

                        if self.scatter_cb.checkState():
                            if self.grad_cb.checkState():
                                l = len(x_val)
                                b = [pg.mkBrush(100, 100, 100, 55 + 200 / l * i) for i in range(l)]
                                p = [pg.mkPen(100, 100, 100, 55 + 200 / l * i) for i in range(l)]
                            else:
                                b = pg.mkBrush(100, 100, 100, 255)
                                p = pg.mkPen(100, 100, 100, 255)
                            self.scatter.setData(x_val, y_val, pen=p, brush=b)

                        if self.lines_cb.checkState():
                            self.lines.setData(x_val, y_val, pen=(0,0,0,100))

                    except:
                        pass

