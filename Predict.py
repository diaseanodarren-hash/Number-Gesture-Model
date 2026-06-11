from ultralytics import YOLO

model = YOLO('Nums_Model3.pt')

#conf = the minimum confidence score the model has that is shown
#source = where the model takes in input ('0' means main webcam; Images can also work, copy the path to your image file) 
#save = if you want to save the result
#PLEASE ADJUST THE DATA ABOVE ACCORDING TO YOUR SYSTEM
results = model(source = 0, conf=0.50, show=True, save=False)[0]
