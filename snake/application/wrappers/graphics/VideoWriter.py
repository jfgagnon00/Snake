
from subprocess import Popen, PIPE
from pygame.surfarray import array3d


class VideoWriter(object):
    """
    Encapsule une GraphicWindow et enregistre son
    rendu dans un fichier video.
    """
    def __init__(self, graphicWindow, fps, filename):
        self._graphicWindow = graphicWindow

        w, h = graphicWindow.size
        self._initSubProcess(filename, fps, w, h)

    def reset(self):
        self._graphicWindow.reset()

    def update(self, gameEnvironment):
        self._graphicWindow.update(gameEnvironment)

    def render(self, message=None):
        self._graphicWindow.render(message)

        image = array3d(self._graphicWindow.surface).swapaxes(0, 1)
        self._writeImage(image)

    def flip(self):
        pass

    def _initSubProcess(self, filename, fps, width, height):
        cmd = [
            "ffmpeg",
            "-loglevel",
            "info",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-y",
            "-c:v",
            "mpeg4",
            "-q:v",
            "0",
            filename,
        ]
        self._subProcess = Popen(cmd, stdin=PIPE)

    def dispose(self):
        self._subProcess.stdin.flush()
        self._subProcess.stdin.close()
        self._subProcess.wait()
        self._subProcess = None

    def _writeImage(self, image):
        self._subProcess.stdin.write(image.tobytes())
