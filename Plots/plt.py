from PymoNNto.Exploration.UI_Base import *

from PymoNNto.Exploration.Evolution.common_UI import *
from PymoNNto.Exploration.Evolution.PlotQTObjects import *

class UI_Plot_Overview(UI_Base):

    def __init__(self, formula, sliders):
        super().__init__(None, label='Plot', create_sidebar=True)

        self.formula = formula

        self.Next_Tab('Plot')

        self.curve = self.Add_plot_curve()

        self.param_list={}
        for s in sliders:
            self.add_slider(s)

    def add_slider(self, param_name):
        self.param_list[param_name] = 0
        slider=self.Add_element(QSlider(Qt.Horizontal), sidebar=True)

        def update_slider(value):
            self.param_list[param_name] = value
            self.update_plot()

        slider.valueChanged.connect(update_slider)

    def update_plot(self):

        global x
        x = np.arange(-10,10,0.01)

        for param, value in self.param_list.items():
            globals()[param] = value

        exec(self.formula, globals())#sets y

        print(y)
        self.curve.setData(x=x, y=y)




########################################################### Exception handling


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)

sys.excepthook = except_hook


if __name__ == '__main__':
    UI_Plot_Overview('''
y1 = x
y = np.sin(x*a)+b + y1
    ''',['a','b']).show()
