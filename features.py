from tkinter import *
def feature(fd):

    root = Tk()
    scrollbar = Scrollbar(root)
    scrollbar.pack( side = RIGHT, fill = Y )
    
    mylist = Listbox(root, yscrollcommand = scrollbar.set )
    for line in range(1,len(fd)):
        mylist.insert(END,fd[line])
    
    mylist.pack( side = LEFT, fill = BOTH )
    scrollbar.config( command = mylist.yview )
    print(len(fd))
    mainloop()       