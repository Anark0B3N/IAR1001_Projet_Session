from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.uix.boxlayout import BoxLayout


def runDrawer():
    MyPaintApp().run()


class MyPaintWidget(Widget):

    def on_touch_down(self, touch):

        with self.canvas:
            Color(1, 1, 1, mode='rgb')
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]


class MyPaintContainerWidget(Widget):
    pass

class MyPaintApp(App):

    def build(self):
        self.parent = Widget()
        self.paint_container = MyPaintContainerWidget()
        self.painter = MyPaintWidget()

        layout = BoxLayout(padding=10, size=(750, 100))
        clearbtn = Button(text='Clear')
        clearbtn.bind(on_release=self.clear_canvas)
        saveBtn = Button(text='save')
        saveBtn.bind(on_release=self.save_canvas)
        layout.add_widget(clearbtn)
        layout.add_widget(saveBtn)

        self.paint_container.add_widget(self.painter)
        self.parent.add_widget(layout)
        self.parent.add_widget(self.paint_container)

        return self.parent

    def clear_canvas(self, obj):
        self.painter.canvas.clear()

    def save_canvas(self, obj):
        self.parent.export_to_png("handwritten_input.png")