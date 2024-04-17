import tkinter as tk
from tkinter.filedialog import askopenfilename

def open_file_dialog(initialdir,title):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = askopenfilename(initialdir=initialdir, title=title)
    return file_path
