from threading import Thread
import cv2 as cv


class WebcamVideoStream:
    '''
        Helper Class For Threaded Image Capturing From Webcam
    '''

    def __init__(self, src=0):

        # TODO :: Remove CAP_DSHOW on Raspberry Pi
        self.video_capture = cv.VideoCapture(src, cv.CAP_DSHOW)

        (self.frame_captured, self.frame) = self.video_capture.read()
        self.stopped = False

    def start(self):

        thread = Thread(target=self.__update, args=())
        thread.daemon = True
        thread.start()
        return self

    def __update(self):
        '''
            Grabs Frames From Video Capture
        '''

        while True:
            # Stop If Necessary
            if self.stopped:
                return

            # Read Next Frame From VideoCapture
            (self.frame_captured, self.frame) = self.video_capture.read()

    def read(self):
        '''
            Returns Whether Frame Was Read And The Most Recent Frame
        '''
        return (self.frame_captured, self.frame)

    def stop(self):
        ''' 
            Stops The Video Capture
        '''
        self.stopped = True
