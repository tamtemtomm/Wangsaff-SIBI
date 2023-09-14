from tkinter import *

class Display():
    def __init__(self):
        pass
    
    def display(self):
        root = Tk()
        
        but = Button(root, text="Click ME!").pack()
        root.mainloop()
    
    def input(self, input):
        pass

main = Display()
main.display()