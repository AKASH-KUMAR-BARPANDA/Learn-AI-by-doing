import cv2

class Camera:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            raise ValueError('Unable to open the camera ')

        self.width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()

    def get_frame(self):
        if self.camera.isOpened():
            is_frame_capture , frame = self.camera.read()

            if is_frame_capture:
                return is_frame_capture,cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            else:
                return is_frame_capture, None
        else:
            return None

