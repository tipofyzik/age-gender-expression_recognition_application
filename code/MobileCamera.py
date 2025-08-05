from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.camera import Camera
from kivy.graphics import PushMatrix, PopMatrix, Rotate
from AttributesPredictor import AttributesPredictor
from kivy.clock import Clock
import os
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



class RotatedCameraWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with self.canvas.before:
            PushMatrix()
            self.rot = Rotate(angle=90, origin=self.center)

        with self.canvas.after:
            PopMatrix()

        self.camera = Camera(index=1, resolution=(640, 480), play=True)
        self.camera.size_hint = (None, None)
        self.add_widget(self.camera)

        self.bind(size=self.update_layout, pos=self.update_layout)
        Clock.schedule_once(self.update_layout, 1)

    def update_layout(self, *args):
        self.rot.origin = self.center
        self.camera.size = (self.height, self.width)
        self.camera.pos = (self.center_x - self.camera.width / 2,
                           self.center_y - self.camera.height / 2)



class MobileCamera(App):
    def build(self):
        self.predictor = AttributesPredictor()
        layout = BoxLayout(orientation='vertical')

        self.camera_widget = RotatedCameraWidget()
        self.camera_widget.size_hint = (1, 0.7)
        layout.add_widget(self.camera_widget)

        btn = Button(text="Take Picture", size_hint=(1, 0.15))
        btn.bind(on_press=self.take_picture)
        layout.add_widget(btn)

        self.label = Label(text="Face attributes will appear here", size_hint=(1, 0.15))
        layout.add_widget(self.label)

        return layout

    def take_picture(self, instance):
        try:
            filepath = os.path.join(self.user_data_dir, "photo.png")
            self.camera_widget.camera.export_to_png(filepath)
            logger.info(f"Photo saved to {filepath}")
            self.process_picture(filepath)
        except Exception as e:
            logger.error("Camera capture failed", exc_info=True)
            self.label.text = f"Camera error: {e}"

    def process_picture(self, filepath):
        try:
            if os.path.exists(filepath):
                result = self.predictor.analyze_image(filepath)
                if result and 'error' in result[0]:
                    self.label.text = result[0]['error']
                else:
                    lines = [f"Age: {r['age']} | Gender: {r['gender']} | Emotion: {r['emotion']}" for r in result]
                    self.label.text = '\n'.join(lines)
            else:
                self.label.text = "Image file not found."
        except Exception as e:
            logger.error("Processing error", exc_info=True)
            self.label.text = f"Processing error: {e}"
