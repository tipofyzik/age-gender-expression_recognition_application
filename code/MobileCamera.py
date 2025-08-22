from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.uix.widget import Widget
from kivy.graphics import PushMatrix, PopMatrix, Rotate, Scale
from kivy.clock import Clock
import os
from PIL import Image

# Android
from jnius import autoclass
from android.permissions import request_permissions, Permission, check_permission

# Custom classes
from AttributesPredictor import AttributesPredictor
from GalleryPicker import GalleryPicker


# Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

Environment = autoclass('android.os.Environment')



class RotatedCameraWidget(Widget):
    """
    A custom camera widget that rotates and mirrors the camera preview.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the camera widget with rotation and mirroring transforms.

        Args:
            **kwargs: Additional keyword arguments for the base Widget class.
        """
        super().__init__(**kwargs)
        self.camera = Camera(index=1, resolution=(1280, 720), play=True)
        self.camera.size_hint = (None, None)
        self.camera.size = (self.width, self.height)
        self.add_widget(self.camera)

        with self.canvas.before:
            PushMatrix()
            self.rot = Rotate(angle=270, origin=self.center)
            self.flip = Scale(x=-1, y=1, z=1, origin=self.center)

        with self.canvas.after:
            PopMatrix()

        self.bind(size=self.update_layout, pos=self.update_layout)
        Clock.schedule_once(self.update_layout, 0)

    def update_layout(self, *args) -> None:
        """
        Updates the layout, position, and transformation origins of the camera
        whenever the widget is resized or moved.

        Args:
            *args: Event arguments passed by Kivy's binding system.

        Returns:
            None
        """
        self.rot.origin = self.center
        self.flip.origin = self.center
        self.camera.size = (self.height, self.width)
        self.camera.pos = (
            self.center_x - self.camera.width / 2,
            self.center_y - self.camera.height / 2
        )


# ------------------------------
# Main app
# ------------------------------
class MobileCamera(App):
    """
    The main Kivy application class for the mobile face attribute analyzer.
    Handles permissions, camera interaction, gallery picking, and attribute prediction.
    """

    def build(self) -> BoxLayout:
        """
        Builds the initial UI of the application with a label placeholder.

        Returns:
            BoxLayout: The root layout of the application.
        """
        self.layout = BoxLayout(orientation='vertical')
        self.label = Label(text="Waiting for permission...", size_hint=(1, 0.3))
        self.layout.add_widget(self.label)
        return self.layout

    def on_start(self) -> None:
        """
        Called when the app starts. Requests camera and storage permissions if needed.

        Returns:
            None
        """
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

    def permission_callback(self, permissions: list[str], grants: list[bool]) -> None:
        """
        Handles the result of a permissions request.

        Args:
            permissions (list[str]): The requested permissions.
            grants (list[bool]): Whether each requested permission was granted.

        Returns:
            None
        """
        if all(grants):
            logger.info("All permissions granted.")
            Clock.schedule_once(lambda dt: self.initialize_ui(), 0)
        else:
            logger.warning("Permissions denied.")
            self.label.text = "Permissions denied. Cannot use camera."

    def initialize_ui(self, *args) -> None:
        """
        Initializes the main user interface after permissions are granted.
        Adds camera widget, buttons, and result label.

        Args:
            *args: Event arguments (ignored).

        Returns:
            None
        """
        self.layout.clear_widgets()

        self.predictor = AttributesPredictor()
        self.camera_widget = RotatedCameraWidget(size_hint=(1, 0.7))
        self.layout.add_widget(self.camera_widget)

        btn_take = Button(text="Take Picture", size_hint=(1, 0.1))
        btn_take.bind(on_press=self.take_picture)
        self.layout.add_widget(btn_take)

        btn_gallery = Button(text="Pick from Gallery", size_hint=(1, 0.1))
        btn_gallery.bind(on_press=self.pick_gallery_image)
        self.layout.add_widget(btn_gallery)

        self.label = Label(text="Face attributes will appear here", size_hint=(1, 0.3))
        self.layout.add_widget(self.label)

    def get_downloads_path(self) -> str:
        """
        Retrieves the absolute path to the Android Downloads directory.

        Returns:
            str: The path to the Downloads folder.
        """
        downloads_dir = Environment.getExternalStoragePublicDirectory(
            Environment.DIRECTORY_DOWNLOADS
        ).getAbsolutePath()
        return downloads_dir

    def save_photo(self, image: Image.Image, filename: str) -> None:
        """
        Saves a given PIL Image to the Downloads folder.

        Args:
            image (Image.Image): The image to save.
            filename (str): The name of the file (e.g., "IMAGE.jpg").

        Returns:
            None
        """
        try:
            downloads_path = self.get_downloads_path()
            filepath = os.path.join(downloads_path, filename)
            image.save(filepath, format="JPEG")
            logger.info(f"Photo saved at {filepath}")
        except Exception as e:
            logger.error("Error saving photo", exc_info=True)
            self.label.text = f"Save error: {e}"

    def pick_gallery_image(self, instance) -> None:
        """
        Opens the gallery picker to allow the user to select an image.

        Args:
            instance: The UI element that triggered this method (e.g., a Button).

        Returns:
            None
        """
        self.gallery_picker = GalleryPicker(self.process_picture)
        self.gallery_picker.pick()

    def take_picture(self, instance) -> None:
        """
        Captures a photo from the camera, rotates it, saves it,
        and processes it for face attribute prediction.

        Args:
            instance: The UI element that triggered this method (e.g., a Button).

        Returns:
            None
        """
        try:
            downloads_path = self.get_downloads_path()
            image_path = os.path.join(downloads_path, "IMAGE.jpg")
            self.camera_widget.camera.export_to_png(image_path)

            # +90° image rotation
            image = Image.open(image_path)
            image = image.rotate(90, expand=True)

            self.save_photo(image, filename="IMAGE.jpg")
            self.process_picture(image_path)
        except Exception as e:
            logger.error("Error capturing photo", exc_info=True)
            self.label.text = f"Camera error: {e}"

    def process_picture(self, filepath: str) -> None:
        """
        Processes a picture: analyzes face attributes and updates the label with results.
        Also saves the cropped face image.

        Args:
            filepath (str): Path to the image file.

        Returns:
            None
        """
        try:
            if os.path.exists(filepath):
                result = self.predictor.analyze_image(filepath)
                cropped_image = self.predictor.return_cropped_face()
                self.save_photo(cropped_image, "FACE.jpg")
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
            logger.error(f"path: {filepath}", exc_info=True)
