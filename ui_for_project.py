# Import Module
import tkinter as tk
import subprocess



# the defs here are the functions for the buttons, here is where
# the instructions to link the other scripts will be done.
def start_button():
    subprocess.run(["python", "simulation_setup.py"])

def image_capture():
    subprocess.run(["python", "image_capture.py"])
def ML_model():
    subprocess.run(["python", "ml_model_runner.py"])
# create root window
root = tk.Tk()

# root window title and dimension
root.title("Farming App")



# all widgets will be here, each button to open each script, as well as one to exit out of the application.

start = tk.Button(root, text="Start Application.", command=start_button)
start.pack()

modelbutton = tk.Button(root, text="ML Model", command=ML_model)
modelbutton.pack()

imagecapture = tk.Button(root, text="Image Capture", command=image_capture)
imagecapture.pack()

turn_off = tk.Button(root, text="Exit Application.", command=root.destroy)
turn_off.pack() 
root.mainloop()

