from jnius import autoclass, cast
from android import activity



Intent = autoclass('android.content.Intent')
PythonActivity = autoclass('org.kivy.android.PythonActivity')
BufferedInputStream = autoclass('java.io.BufferedInputStream')



class GalleryPicker:
    """
    A helper class for picking an image from the Android gallery
    and returning its file path to a callback function.
    """

    REQUEST_CODE = 1001

    def __init__(self, on_complete_callback: callable = None) -> None:
        """
        Initializes the gallery picker and binds the activity result listener.

        Args:
            on_complete_callback (callable | None): A function to be called when the image
                selection is complete. It will receive either the file path (str) or None.
        """
        self.on_complete = on_complete_callback
        activity.bind(on_activity_result=self._on_activity_result)

    def pick(self) -> None:
        """
        Opens the Android gallery so the user can pick an image.

        Returns:
            None
        """
        intent = Intent(Intent.ACTION_PICK)
        intent.setType("image/*")
        currentActivity = cast('android.app.Activity', PythonActivity.mActivity)
        currentActivity.startActivityForResult(intent, self.REQUEST_CODE)

    def get_file_path_from_uri(self, uri) -> str | None:
        """
        Attempts to resolve an absolute file path from a given Android content URI.

        Args:
            uri: Android URI object (content://).

        Returns:
            str | None: The absolute file path if available, otherwise None.
        """
        try:
            context = PythonActivity.mActivity
            resolver = context.getContentResolver()
            MediaStoreMediaColumns = autoclass('android.provider.MediaStore$MediaColumns')
            proj = [MediaStoreMediaColumns.DATA]
            cursor = resolver.query(uri, proj, None, None, None)

            if cursor is None:
                return None

            column_index = cursor.getColumnIndexOrThrow(proj[0])
            cursor.moveToFirst()
            file_path = cursor.getString(column_index)
            cursor.close()

            return file_path
        except Exception as e:
            print(f"[GalleryPicker] Error getting real path: {e}")
            return None

    def _on_activity_result(self, requestCode: int, resultCode: int, intent) -> None:
        """
        Callback method triggered when the gallery activity returns a result.

        Args:
            requestCode (int): The request code used to identify the activity result.
            resultCode (int): The result status code returned by the activity.
            intent: The intent returned from the gallery activity.

        Returns:
            None
        """
        if requestCode == self.REQUEST_CODE:
            if intent is None:
                print("[GalleryPicker] Выбор отменён")
                if self.on_complete:
                    self.on_complete(None)
                return

            uri = intent.getData()
            file_path = self.get_file_path_from_uri(uri)

            if not file_path:
                print("[GalleryPicker] Не удалось получить путь к файлу")
                if self.on_complete:
                    self.on_complete(None)
                return

            if self.on_complete:
                self.on_complete(file_path)