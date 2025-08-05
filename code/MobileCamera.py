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

from jnius import autoclass, cast
from android.permissions import request_permissions, Permission, check_permission
from android import activity



from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.uix.widget import Widget
from kivy.graphics import PushMatrix, PopMatrix, Rotate
from kivy.clock import Clock
import os
import logging

# Только на Android
from android.permissions import request_permissions, Permission, check_permission

from AttributesPredictor import AttributesPredictor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RotatedCameraWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = Camera(index=1, resolution=(640, 480), play=True)
        self.camera.size_hint = (None, None)
        self.camera.size = (self.width, self.height)
        self.add_widget(self.camera)

        with self.canvas.before:
            PushMatrix()
            self.rot = Rotate(angle=90, origin=self.center)

        with self.canvas.after:
            PopMatrix()

        self.bind(size=self.update_layout, pos=self.update_layout)
        Clock.schedule_once(self.update_layout, 0)

    def update_layout(self, *args):
        self.rot.origin = self.center
        self.camera.size = (self.height, self.width)
        self.camera.pos = (
            self.center_x - self.camera.width / 2,
            self.center_y - self.camera.height / 2
        )


class MobileCamera(App):
    def build(self):
        self.permission_granted = False
        self.layout = BoxLayout(orientation='vertical')
        self.label = Label(text="Waiting for permission...", size_hint=(1, 0.3))
        self.layout.add_widget(self.label)
        return self.layout

    def on_start(self):
        needed = [
            Permission.CAMERA,
            Permission.WRITE_EXTERNAL_STORAGE,
            Permission.READ_EXTERNAL_STORAGE
        ]

        if all(check_permission(p) for p in needed):
            logger.info("Permissions already granted.")
            Clock.schedule_once(lambda dt: self.initialize_ui(), 0)
        else:
            logger.info("Requesting permissions...")
            request_permissions(needed, self.permission_callback)

    def permission_callback(self, permissions, grants):
        if all(grants):
            logger.info("All permissions granted.")
            Clock.schedule_once(lambda dt: self.initialize_ui(), 0)
        else:
            logger.warning("Permissions denied.")
            self.label.text = "Permissions denied. Cannot use camera."

    def initialize_ui(self, *args):
        self.layout.clear_widgets()

        self.predictor = AttributesPredictor()

        # Камера
        self.camera_widget = RotatedCameraWidget(size_hint=(1, 0.7))
        self.layout.add_widget(self.camera_widget)

        # Кнопка
        btn = Button(text="Take Picture", size_hint=(1, 0.2))
        btn.bind(on_press=self.take_picture)
        self.layout.add_widget(btn)

        # Метка результата
        self.label = Label(text="Face attributes will appear here", size_hint=(1, 0.3))
        self.layout.add_widget(self.label)

    def take_picture(self, instance):
        try:
            filepath = os.path.join(self.user_data_dir, "photo.png")
            self.camera_widget.camera.export_to_png(filepath)
            logger.info(f"Photo saved to {filepath}")
            self.process_picture(filepath)
        except Exception as e:
            logger.error("Error taking picture", exc_info=True)
            self.label.text = f"Camera error: {e}"

    def process_picture(self, filepath):
        try:
            if os.path.exists(filepath):
                result = self.predictor.analyze_image(filepath)
                if result and isinstance(result, list) and 'error' in result[0]:
                    self.label.text = result[0]['error']
                else:
                    lines = [
                        f"Age: {f.get('age', '?')} | Gender: {f.get('gender', '?')} | Emotion: {f.get('emotion', '?')}"
                        for f in result
                    ]
                    self.label.text = '\n'.join(lines)
            else:
                self.label.text = "Picture not found."
        except Exception as e:
            logger.error("Error processing picture", exc_info=True)
            self.label.text = f"Processing error: {e}"

