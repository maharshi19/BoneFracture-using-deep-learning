import tkinter
from tkinter import filedialog as fd
from PIL import ImageTk, Image 
from tkinter import ttk 
import cv2
import numpy as np
master = tkinter.Tk() 
master.configure(bg='#347286') #https://cssgradient.io/
master.title("BONE FACTURE CLASSIFICATION USING DEEP LEARNING")
var = tkinter.StringVar()
label = tkinter.Label( master, textvariable=var, fg = "Darkblue",bg = "LightGreen",font = "Verdana 15 bold")
var.set("BONE FACTURE CLASSIFICATION USING DEEP LEARNING")
label.pack()

def BowseImage():
    name= fd.askopenfilename()
    img = Image.open(name) 
    img.save("Model/query.jpg")
    img = img.resize((250, 250), Image.ANTIALIAS)   
    img = ImageTk.PhotoImage(img)    
    panel = tkinter.Label(master, image = img)       
    panel.image = img 
    panel.place(x = 180, y = 5 + 1*30, width=250, height=250)
    

def Preprocess():
    img = cv2.imread("Model/query.jpg")
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    argument=op.get()
    if argument=="Median":
        img=cv2.medianBlur(img, 3)
    if argument=="Gaussian":
        img=cv2.GaussianBlur(img,(5,5),0)
    cv2.imwrite("Model/Pre.jpg",img)
    img = Image.open("Model/Pre.jpg") 
    img = img.resize((250, 250), Image.ANTIALIAS) 
    img = ImageTk.PhotoImage(img)   
    panel = tkinter.Label(master, image = img)       
    panel.image = img 
    panel.place(x = 460, y = 5 + 1*30, width=250, height=250)
        
    master.mainloop()
def Segment():
    img = cv2.imread("Model/Pre.jpg",0)
    Z = img.reshape((-1,2))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    ret, res2 = cv2. threshold(res2,res2.max()-15,255,cv2.THRESH_BINARY)
    cv2.imwrite("Model/Seg.jpg",res2)
    img = Image.open("Model/Seg.jpg") 
    img = img.resize((250, 250), Image.ANTIALIAS) 
    img = ImageTk.PhotoImage(img)   
    panel = tkinter.Label(master, image = img)       
    panel.image = img 
    panel.place(x = 180, y = 265 + 1*30, width=250, height=250)
    master.mainloop()

def feature():
    #Shape
    from skimage.measure import label, regionprops
    res2 = cv2.imread("Model/Seg.jpg")
    label_img = label(res2)
    areas = min([r.area for r in regionprops(label_img)])
    centroid = min([r.centroid for r in regionprops(label_img)])
    
    #Texture
    img = cv2.imread("Model/Pre.jpg",0)
    from skimage.feature import greycomatrix, greycoprops
    g = greycomatrix(img, [1], [0],  symmetric = True, normed = True )
    contrast = greycoprops(g, 'contrast')
    correlation = greycoprops(g, 'correlation')
    energy = greycoprops(g, 'energy')
    homogeneity = greycoprops(g, 'homogeneity')
    
    #HOG
    img = cv2.imread("Model/Pre.jpg")
    from skimage.feature import hog
    from skimage.transform import resize
    resized_img = resize(img, (128*4, 64*4))
    fh, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(64, 64),cells_per_block=(2, 2), visualize=True, multichannel=True)
    
    ft1=[areas,centroid[0],float(contrast),float(correlation),float(energy),float(homogeneity)]
    for i in fh : 
        ft1.append(i)
    ft1 = [ '%.2f' % elem for elem in ft1 ]
    
    #GWT
    img = cv2.imread("Model/Pre.jpg",0)
    from scipy import ndimage as ndi
    from skimage.filters import gabor_kernel
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    GBW = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(img, kernel, mode='wrap')
        GBW[k, 0] = filtered.mean()
        GBW[k, 1] = filtered.var()
    GBWF=GBW.reshape((-1, 1))
    ft2=[areas,centroid[0],float(contrast),float(correlation),float(energy),float(homogeneity)]
    for i in GBWF : 
        ft2.append(i)
    ft2 = [ '%.2f' % elem for elem in ft2 ]

    argument=op1.get()
    if argument=="GLCM+HOG+Shape":
        import features
        features.feature(ft1)
    if argument=="GLCM+GWT+shape":
        import features
        features.feature(ft2)
    master.mainloop()   

def Test():
    img11=cv2.imread("Model/query.jpg")
    img11 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
    img11 = cv2.resize(img11, (64,64), interpolation = cv2.INTER_AREA)
    img11 = cv2.medianBlur(img11, 3)
    query = img11.reshape((1, -1))
    import pickle
    argument=op2.get()
    if argument=="SVM":  
        loaded_model = pickle.load(open("Model/SVM.sav", 'rb'))
        predicted = loaded_model.predict(query)
        AC="76.78%"
        PR="73.42%"
        RC="74.23%"
    if argument=="RF":  
        loaded_model = pickle.load(open("Model/RF.sav", 'rb'))
        predicted = loaded_model.predict(query)
        AC="82.23%"
        PR="83.34%"
        RC="83.45%"
    if argument=="DT":  
        loaded_model = pickle.load(open("Model/DT.sav", 'rb'))
        predicted = loaded_model.predict(query)
        AC="85.65"
        PR="84.84"
        RC="84.78"
    if argument=="KNN":  
        loaded_model = pickle.load(open("Model/KNN.sav", 'rb'))
        predicted = loaded_model.predict(query)
        AC="78.76%"
        PR="77.45%"
        RC="77.48%"
    if argument=="DNN":
        from keras.models import load_model
        import numpy as np
        import warnings
        warnings.filterwarnings("ignore")
        model = load_model('Model/model.h5')
        fr = np.array(img11)
        fr = fr.reshape(1, 64, 64, 1)
        fr = fr.astype('float32') / 255
        clas=model.predict_classes(fr)
        predicted=[]
        if int(clas)==0 :
            predicted.append('XR_SHOULDER_positive')
        if int(clas)==1 :
            predicted.append('XR_SHOULDER_negative')
        if int(clas)==2 :
            predicted.append('XR_HUMERUS_positive')
        if int(clas)==3 :
            predicted.append('XR_HUMERUS_negative')
        if int(clas)==4 :
            predicted.append('XR_FINGER_positive')
        if int(clas)==5 :
            predicted.append('XR_FINGER_negative')
        if int(clas)==6 :
            predicted.append('XR_ELBOW_positive') 
        if int(clas)==7 :
            predicted.append('XR_ELBOW_negative')
        if int(clas)==8 :
            predicted.append('XR_WRIST_positive')
        if int(clas)==9 :
            predicted.append('XR_WRIST_negative') 
        if int(clas)==10 :
            predicted.append('XR_FOREARM_positive')
        if int(clas)==11 :
            predicted.append('XR_FOREARM_negative')
        if int(clas)==12 :
            predicted.append('XR_HAND_positive')
        if int(clas)==13 :
            predicted.append('XR_HAND_negative')
        AC="87.94%"
        PR="86.75%"
        RC="86.63%"
    
    var = tkinter.StringVar()
    label = tkinter.Label( master, textvariable=var, fg = "Red",bg = "white",font = "Verdana 8 bold")
    var.set(str(predicted[0]))
    label.place(x = 10, y = 230 + 10*30, width=150, height=50)
    var1 = tkinter.StringVar()
    label = tkinter.Label( master, textvariable=var1, fg = "Red",bg = "white",font = "Verdana 10 bold")
    var1.set(AC)
    label.place(x = 10, y = 255 + 11*30, width=150, height=50)  
    var2 = tkinter.StringVar()
    label = tkinter.Label( master, textvariable=var2, fg = "Red",bg = "white",font = "Verdana 10 bold")
    var2.set(PR)
    label.place(x = 10, y = 280 + 12*30, width=150, height=50)  
    var3 = tkinter.StringVar()
    label = tkinter.Label( master, textvariable=var3, fg = "Red",bg = "white",font = "Verdana 10 bold")
    var3.set(RC)
    label.place(x = 10, y = 305 + 13*30, width=150, height=50)
    master.mainloop()


master.geometry("750x750+100+100") 
master.resizable(width = True, height = True) 

b1 = tkinter.Button(master, text = "Bowse Image", command = BowseImage,bg='#505057',fg='white',font = "Verdana 10 bold") 
b1.place(x = 10, y = 5 + 1*30, width=150, height=50)

op = ttk.Combobox(master,values=["Median","Gaussian",],font = "Verdana 10 bold")
op.place(x = 10, y = 30 + 2*30, width=150, height=50)
op.current(0)

b2 = tkinter.Button(master, text = "Pre-Process", command = Preprocess,bg='#505057',fg='white',font = "Verdana 9 bold") 
b2.place(x = 10, y = 55 + 3*30, width=150, height=50)

b3 = tkinter.Button(master, text = "Segmentation", command = Segment,bg='#505057',fg='white',font = "Verdana 9 bold") 
b3.place(x = 10, y = 80 + 4*30, width=150, height=50)

op1 = ttk.Combobox(master,values=["GLCM+HOG+Shape","GLCM+GWT+shape"],font = "Verdana 10 bold")
op1.place(x = 10, y = 105 + 5*30, width=150, height=50)
op1.current(0)

b4 = tkinter.Button(master, text = "Extract Feature", command = feature,bg='#505057',fg='white',font = "Verdana 9 bold") 
b4.place(x = 10, y = 130 + 6*30, width=150, height=50)

op2 = ttk.Combobox(master,values=["SVM","RF","DT","KNN","DNN"],font = "Verdana 10 bold")
op2.place(x = 10, y = 155 + 7*30, width=150, height=50)
op2.current(0)


b5 = tkinter.Button(master, text = "Classify", command = Test,bg='#505057',fg='white',font = "Verdana 10 bold") 
b5.place(x = 10, y = 180 + 8*30, width=150, height=50)


var = tkinter.StringVar()
label = tkinter.Label( master, textvariable=var, fg = "Red",bg = "white",font = "Verdana 10 bold")
var.set("Facture Type")
label.place(x = 10, y = 205 + 9*30, width=150, height=50)

var1 = tkinter.StringVar()
label = tkinter.Label( master, textvariable=var1, fg = "Red",bg = "white",font = "Verdana 10 bold")
var1.set("Accuracy")
label.place(x = 10, y = 230 + 10*30, width=150, height=50)


var2 = tkinter.StringVar()
label = tkinter.Label( master, textvariable=var2, fg = "Red",bg = "white",font = "Verdana 10 bold")
var2.set("Precision")
label.place(x = 10, y = 255 + 11*30, width=150, height=50)

var3 = tkinter.StringVar()
label = tkinter.Label( master, textvariable=var3, fg = "Red",bg = "white",font = "Verdana 10 bold")
var3.set("Recall")
label.place(x = 10, y = 280 + 12*30, width=150, height=50)


master.mainloop() 

