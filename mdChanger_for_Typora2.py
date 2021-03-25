
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from os import listdir
from os.path import isfile, join
import os
import shutil
import time

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 300, height = 250, bg = 'white', relief = 'raised')
canvas1.pack()

label1 = tk.Label(root, text='md-changer\nfor-typoraImage', bg = 'white')
label1.config(font=('helvetica', 20))
canvas1.create_window(150, 60, window=label1)


link1 = "C:\\Users\\myunghak\\AppData\\Roaming\\Typora\\typora-user-images\\"
link2 = "https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/"


def getFile ():
    temp_link = "file:///C:/Users/myunghak/AppData/Local/Temp/msohtmlclip1/"

    read_path = ""
    write_path = ""
    folder_name = str(time.time()).replace(".","_")
    try:
        folder = os.listdir(temp_link[8:])
        print(len(folder))
        assert len(folder) < 2, "The folder is wrong"


        for f in folder:
            print(f)
            print("=" * 100)
            print(f"copy folder : {temp_link[8:] + f} ----------> {link1 + folder_name}")
            shutil.copytree(temp_link[8:] + f, link1 + folder_name)
    except FileNotFoundError:
        pass


    read_path = filedialog.askopenfilename()
    print("Read : "+ read_path)

    read_old_path = read_path[:-3] + "-old" + read_path[-3:]
    os.rename(read_path, read_old_path)
    write_path = read_path
    
    out = open(write_path, 'wt', encoding='UTF8')
    with open(read_old_path,'rt', encoding='UTF8') as fp:
        while True:
            line = fp.readline()
            if not line: break

                
            line1 = line.replace(temp_link + folder[0] + "/",link2 + folder_name + "/")
            line2 = line1.replace(".png",".png?raw=tru")


            line3 = line2.replace(link1,link2)
            line4 = line3.replace(".png",".png?raw=tru")
            out.write(line4)
    out.close()
    print("Saved md-changed file completely ")
    
browseButton = tk.Button(text="Select & Change", command=getFile, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 130, window=browseButton)

def exitApplication():
    MsgBox = tk.messagebox.askquestion ('Exit Application','Are you sure you want to exit the application',icon = 'warning')
    if MsgBox == 'yes':
       root.destroy()
     
exitButton = tk.Button (root, text='Exit Application',command=exitApplication, bg='brown', fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 180, window=exitButton)

root.mainloop()



