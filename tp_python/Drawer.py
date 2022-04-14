import cv2
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.uix.boxlayout import BoxLayout
import numpy
from kivy.config import Config

import CNN
from ImageDetails import IMG_DETAILS

Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '500')
Config.set('graphics', 'height', '610')


def runDrawer():
    MyPaintApp().run()


class MyPaintWidget(Widget):

    def on_touch_down(self, touch):

        with self.canvas:
            Color(1, 1, 1, mode='rgb')
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=17)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]


class MyPaintContainerWidget(Widget):
    pass

class MyPaintApp(App):

    def build(self):
        self.parent = Widget()
        self.paint_container = MyPaintContainerWidget()
        self.painter = MyPaintWidget()

        layout = BoxLayout(padding=10, size=(500, 100))
        clearbtn = Button(text='Clear')
        clearbtn.bind(on_release=self.clear_canvas)
        saveBtn = Button(text='save')
        saveBtn.bind(on_release=self.save_canvas)
        label = Label(text='This number is: X', font_size='20sp')
        layout.add_widget(clearbtn)
        layout.add_widget(saveBtn)
        layout.add_widget(label)

        self.paint_container.add_widget(self.painter)
        self.parent.add_widget(layout)
        self.parent.add_widget(self.paint_container)

        return self.parent

    def clear_canvas(self, obj):
        self.painter.canvas.clear()

    def save_canvas(self, obj):
        self.parent.export_to_png("handwritten_input.png")
        savedImg = cv2.imread("handwritten_input.png", cv2.COLOR_BGRA2GRAY)


        savedImg = cv2.cvtColor(savedImg, cv2.COLOR_BGRA2GRAY)
        savedImg = savedImg[0:len(savedImg) - 110, 0:len(savedImg[0])]
        savedImg = cv2.resize(savedImg, (IMG_DETAILS.IMG_SIZE, IMG_DETAILS.IMG_SIZE))

        mass_center_x = 0
        mass_center_count = 0
        mass_center_y = 0

        x = 0
        for row in savedImg:
            y = 0
            for pixel in row:
                if pixel > 51:
                    mass_center_x += x
                    mass_center_y += y
                    mass_center_count += 1
                y += 1

            x += 1

        mass_center_x /= mass_center_count
        mass_center_y /= mass_center_count

        imgCenter = int(IMG_DETAILS.IMG_SIZE / 2)

        savedImg = translateImg(savedImg, int(imgCenter - mass_center_x), int(imgCenter - mass_center_y))

        cv2.imwrite("handwritten_input.png", savedImg)

        CNN.test_model()


def translateImg(img, x_transl, y_transl):
    toReturn = numpy.zeros((IMG_DETAILS.IMG_SIZE,IMG_DETAILS.IMG_SIZE))
    x = 0
    for row in img:
        y = 0
        for pixel in row:
            if pixel > 51:
                # toReturn[x][y] = 0
                x_pos = x + x_transl
                y_pos = y + y_transl
                if x_pos > -1 and y_transl > -1 and x_pos < IMG_DETAILS.IMG_SIZE and y_pos < IMG_DETAILS.IMG_SIZE:
                    toReturn[x_pos][y_pos] = 255
                else:
                    a = 8

            y += 1
        x += 1

    return toReturn