from ultralytics import YOLO

model = YOLO('Nums_Model3.pt')

results = model(source = 0, conf=0.50, show=True)[0]