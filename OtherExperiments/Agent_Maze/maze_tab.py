from PymoNNto.Exploration.Network_UI.TabBase import *
from OtherExperiments.Agent_Maze.Maze import *

class Maze_Draw_Item(pg.GraphicsObject):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.picture = QtGui.QPicture()
        #self.update()


    def update_pic(self, maze):

        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)

        painter.setPen(pg.mkPen(color=(0, 0, 0, 255)))
        painter.setBrush(pg.mkBrush(color=(255, 255, 255, 255)))

        painter.drawRect(QtCore.QRectF(0, 0, maze.maze_w, -maze.maze_h))


        for box in maze.boxes:
            painter.setPen(pg.mkPen(color=box.color))
            painter.setBrush(pg.mkBrush(color=box.color))
            painter.drawRect(QtCore.QRectF(box.x, -box.y, box.w, -box.h))

        painter.setPen(pg.mkPen(color=(255, 0, 0, 255)))

        for ray in maze.rays:
            painter.setPen(pg.mkPen(color=(255, 0, 0, 255)))
            painter.setBrush(pg.mkBrush(color=(255, 0, 0, 255)))
            if maze.collision_ray_color:
                if ray.collision_box is not None:
                    c = ray.collision_box.color
                    if c is not None:
                        cf = 1.0#(7-d)/7
                        painter.setPen(pg.mkPen(color=(c[0]*cf, c[1]*cf, c[2]*cf, 255)))
                        painter.setBrush(pg.mkBrush(color=(c[0]*cf, c[1]*cf, c[2]*cf, 255)))
            painter.drawLine(QtCore.QLineF(ray.x, -ray.y, ray.x + ray.dx * ray.dist, -ray.y - ray.dy * ray.dist))

            #painter.drawRect(QtCore.QRectF(-10+ray.dx*5, -5-ray.dy*5, 1, 1))

        painter.setPen(pg.mkPen(color=maze.player.color))
        painter.setBrush(pg.mkBrush(color=maze.player.color))
        painter.drawRect(QtCore.QRectF(maze.player.x, -maze.player.y, box.w, -box.h))

        painter.setPen(pg.mkPen(color=maze.goal.color))
        painter.setBrush(pg.mkBrush(color=maze.goal.color))
        painter.drawRect(QtCore.QRectF(maze.goal.x, -maze.goal.y, box.w, -box.h))

        painter.end()
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        #self.update()

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        br=self.picture.boundingRect()
        br.setTop(min(br.top(), br.left()))
        br.setLeft(min(br.top(), br.left()))
        br.setRight(max(br.right(), br.bottom()))
        br.setBottom(max(br.right(), br.bottom()))
        return QtCore.QRectF(br)

class maze_tab(TabBase):

    def __init__(self, title='Maze'):
        super().__init__(title)
        self.sidebar = True

    def add_recorder_variables(self, neuron_group, Network_UI):
        return

    def initialize(self, Network_UI):
        if hasattr(Network_UI.network, 'maze'):
            if not self.sidebar:
                self.mazetab = Network_UI.Next_Tab(self.title)

            self.plot = Network_UI.Add_plot(sidebar=self.sidebar)
            self.draw_item = Maze_Draw_Item()
            self.plot.addItem(self.draw_item)

            def up(event):
                Network_UI.network.maze.move_player(0, -1)

            up_btn = QPushButton('up', Network_UI.main_window)
            up_btn.clicked.connect(up)
            Network_UI.Add_element(up_btn, sidebar=self.sidebar)

            def down(event):
                Network_UI.network.maze.move_player(0, 1)

            down_btn = QPushButton('down', Network_UI.main_window)
            down_btn.clicked.connect(down)
            Network_UI.Add_element(down_btn, sidebar=self.sidebar)

            def left(event):
                Network_UI.network.maze.move_player(-1, 0)

            left_btn = QPushButton('left', Network_UI.main_window)
            left_btn.clicked.connect(left)
            Network_UI.Add_element(left_btn, sidebar=self.sidebar)

            def right(event):
                Network_UI.network.maze.move_player(1, 0)

            right_btn = QPushButton('right', Network_UI.main_window)
            right_btn.clicked.connect(right)
            Network_UI.Add_element(right_btn, sidebar=self.sidebar)


    def update(self, Network_UI):
        if hasattr(Network_UI.network, 'maze'):
            self.draw_item.update_pic(Network_UI.network.maze)
            self.plot.update()

            Network_UI.network.maze.random_move()

