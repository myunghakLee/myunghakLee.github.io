
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import shutil

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 300, height = 250, bg = 'white', relief = 'raised')
canvas1.pack()

label1 = tk.Label(root, text='md-changer\nfor-typoraImage', bg = 'white')
label1.config(font=('helvetica', 20))
canvas1.create_window(150, 60, window=label1)

link1 = "C:\\Users\\mh9716\\AppData\\Roaming\\Typora\\typora-user-images\\"
link2 = "https://github.com/myunghakLee/myunghakLee.github.io/blob/main/blog_images/"
def getFile ():
    read_path = ""
    write_path = ""
    read_path = filedialog.askopenfilename()
    print("Read : "+ read_path)
    shutil.copy2(read_path, read_path + ".tmp")
    write_path = read_path[:-3] + read_path[-3:]
    out = open(write_path, 'wt', encoding='UTF8')
    with open(read_path+ ".tmp",'rt', encoding='UTF8') as fp:
        while True:
            line = fp.readline()
            print(line)
            if not line: break

            if "typora-user-images" in line:
                file_name = line.split("\\")[-1][:-1]
                while (file_name[-1] == ')'):
                    file_name = file_name[:-1]                
                print((link1 + file_name).replace("\\\\", "\\"))
                shutil.copy2(link1 + file_name, "/gitpage/myunghakLee.github.io/blog_images/")

            line1 = line.replace(link1,link2)
            line2 = line1.replace(".png",".png?raw=tru")
            print(line2)
            out.write(line2)
    out.close()
    print(link1)
    print(link2)

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



