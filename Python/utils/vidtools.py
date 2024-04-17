import cv2 as cv
import numpy as np
import time as tm
import os
import PySpin as ps
import tkinter as tk
from PIL import Image, ImageTk
import tifffile


class CameraApp:
    '''This is code written with chatGPT to get a live feed from a passed camera object
    Michael Reitman 2024/01/08 '''
    def __init__(self, camera):
        self.camera = camera
        self.root = tk.Tk()
        self.root.title("Live Camera Feed")
        self.label = tk.Label(self.root)
        self.label.pack()

        #set initial gui size
        # Get screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calculate window size
        window_width =  int(screen_width * 0.5)
        window_height =  int(screen_height * 0.8)

        # Set window size and position (centered)
        x = (screen_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f'{window_width}x{window_height}+{x}+{y}')

        # Start the camera acquisition
        self.camera.BeginAcquisition()
        self.update_image()

        #initialize variables for smooth closing
        self.after_id = None  # Initialize a variable to store the 'after' call ID
        self.is_closed = False  # Flag to indicate if the window is already closed

        # Bind the escape key and window close event to the close method
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def update_image(self):
        try:
            # Capture image
            image_result = self.camera.GetNextImage()

            if image_result.IsIncomplete():
                print("Image incomplete with image status %d ..." % image_result.GetImageStatus())
            else:
                # Convert to a format suitable for display
                image_data= image_result.GetNDArray()

                # Convert to PIL Image
                pil_image = Image.fromarray(image_data)
                tk_image = ImageTk.PhotoImage(pil_image)

                # Resize the image to fit the window
                window_width = self.root.winfo_width()
                window_height = self.root.winfo_height()

                # Check if window dimensions are valid (non-zero)
                if window_width > 0 and window_height > 0:
                    pil_image = Image.fromarray(image_data)
                    resized_image = pil_image.resize((window_width, window_height))

                    tk_image = ImageTk.PhotoImage(resized_image)

                # Update the label with the new image
                self.label.configure(image=tk_image)
                self.label.image = tk_image

            # Release the image
            image_result.Release()

            # Update the image every 100ms
            self.after_id =  self.root.after(100, self.update_image)

        except ps.SpinnakerException as ex:
            print("Error: %s" % ex)

    def close_event(self, event):
        self.close()

    def close(self):
        #Check if the window is already closed to avoid repeating the cleanup
        if self.is_closed:
            return

        # Mark the window as closed
        self.is_closed = True

        # End camera acquisition
        if self.camera.IsStreaming():
            self.camera.EndAcquisition()
                # Cancel the scheduled 'after' call if it exists

        if self.after_id is not None:
            self.root.after_cancel(self.after_id)

        self.root.destroy()

    def run(self):
        self.root.mainloop()

def detect_cams(n=None):
    """From Dave Mets Camtools. detects cameras.  Helpful to do before trying to change anything!"""
    if n is None:
        n=1
    sys = ps.System.GetInstance()
    cam_list=sys.GetCameras()
    n_cams=cam_list.GetSize()
    if n_cams<n:
        print('Not enough cameras detected!')
        cam_list.Clear()
        sys.ReleaseInstance()
        return False
    else:
        cam_list.Clear()
        sys.ReleaseInstance()
        return True

def load_video(video_path, display=False):

    cap = cv.VideoCapture(video_path)
    images = list()

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            images.append(frame)
            if display:
                cv.imshow(video_path.split('/')[-1],frame)
                cv.waitKey(1)
        else:
            break

    cap.release()

    return images

def save_video(images,path='./',fourcc_code=0, ext='.avi',prefix='videofile',
               barcode='00000',frame_rate=160.0,is_color=False):
    '''Modified from Dave Mets. OpenCV utility to save a video.  It expects 'images' to be a numpy array'''

    if not os.path.isdir(path):
        os.makedirs(path)

    tme=int(tm.time())
    im_height=len(images[0])
    im_width=len(images[0][0])
    filename=path+prefix+'_'+str(barcode)+'_'+str(tme)+ext

    video_handler=cv.VideoWriter(filename,fourcc_code,frame_rate,(im_width,im_height),is_color)
    for image in images:
        video_handler.write(image)

    cv.destroyAllWindows()
    video_handler.release()

def write_tiff_stack(arrays,output_path,filename,data_type='uint8'):
    """
    ChatGPT code generated by MR on 2024_02_09 to write a list of uint16 arrays to a TIFF stack.

    Parameters:
        arrays (list of numpy arrays): List of uint16 arrays to be written to the TIFF stack.
        filepath (str): Path to save the TIFF stack.
    """
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Convert the list of arrays to a numpy stack
    stack = np.stack(arrays)

    # Write the stack to a TIFF file
    with tifffile.TiffWriter(filename, bigtiff=True) as tif:
        for img in stack:
            if data_type=='uint8':
                tif.save(img.astype(np.uint8))
            if data_type=='uint16':
                tif.save(img.astype(np.uint16))

def grab_images(cam, node_map, length=None, n_frames=None,data_type='uint8'):
    """Modified from Dave Mets. this grabs a set of images from a camera.  It assumes the camera has been initialized and
    all desired changes to acquisition have been made on the camera. Expects 'length' to be in
    seconds"""
    if length==None and n_frames==None:
        n_frames=1
    elif not length==None and n_frames==None:
        frame_rate= cam.AcquisitionFrameRate.GetValue()
        exposure_time=cam.ExposureTime()
        if 1000**2/exposure_time<frame_rate:
            frame_rate=1000**2/exposure_time
        n_frames=int(float(length)*frame_rate)
    elif not length==None and not n_frames==None:
        print('plese specify either the number of frames or the length of the acquisition not both')
        return

    host_timestamps=[]
    images=n_frames*[None]
    dropped_frames=[]
    node_map = cam.GetTLStreamNodeMap()
    frame_ids = []
    cam_timestamps = []
    active_sequence = []

    try:
        cam.BeginAcquisition()
        timeout=1000
        for i in range(n_frames):
            curr_time=tm.time()
            image=cam.GetNextImage(timeout)
            host_timestamps.append(curr_time)

            # images[i] = image.GetNDArray()
            images[i] = np.array(image.GetNDArray(), dtype=data_type)

            #record if any frames got dropped from the buffer
            dropped_frames.append(ps.CIntegerPtr(node_map.GetNode('StreamDroppedFrameCount')).GetValue()) #keep a running tally of any dropped frames
            ## get and store metadata
            chunk_data = image.GetChunkData()
            # Retrieve frame ID
            frame_ids.append(chunk_data.GetFrameID())
            active_sequence.append(chunk_data.GetSequencerSetActive())
            cam_timestamps.append(chunk_data.GetTimestamp())
            image.Release()
        cam.AcquisitionStop()

        return images,host_timestamps,dropped_frames,frame_ids,active_sequence,cam_timestamps
    except ps.SpinnakerException as ex:
        cam.EndAcquisition()
        print('Error: %s' % ex)
        return False

def z_profile(video):
    z_profile = []
    for frame in video:
        z_profile.append(np.mean(frame))

    return z_profile

def load_z_profile(video_path):

    cap = cv.VideoCapture(os.path.join(video_path))
    z_profile = []
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            z_profile.append(np.mean(frame))

        else:
            break

    cap.release()

    return z_profile
