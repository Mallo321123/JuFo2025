#As an input an image of cars waiting at a intersection is used.
#(This image will be replaced with the live feed of the webcam.)
#The output consists of the number of detected cars. 

from ultralytics import YOLO

#Used model
model = YOLO("yolov8n.pt")

#For objective testing I ran an collection of images in a folder through the model
#For this I collected different images from my way to school and also a road constuction:
#results = model(source="/home/lwcbwhite/TrafficAid/object_recognition/YOLOv8/car", show=False, conf=0.23, save=True)

#Class identifier(COCO dataset): 1 --> bicycle, 2 --> car, 3 --> motorcycle, 5 --> bus, 7 --> truck

def counting_vehicles(image_path):
    #Running the model on the image with a threshold of 23%
    #Threshold was determined through the F1-confidence curve as well as 
    #the precison-confidence curve and the recall-confidence curve
    results = model(image_path, show=False, conf=0.23, save=True)
    detections = results[0]

    #Extracting the numbers of detected objects per class in the reference frame 
    bicycle_detections = [det for det in detections.boxes if det.cls == 1]
    num_bicycles = len(bicycle_detections)
    car_detections = [det for det in detections.boxes if det.cls == 2]
    num_car = len(car_detections)
    mcycle_detections = [det for det in detections.boxes if det.cls == 3]
    num_mcycle= len(mcycle_detections)
    bus_detections = [det for det in detections.boxes if det.cls == 5]
    num_bus = len(bus_detections)
    truck_detections = [det for det in detections.boxes if det.cls == 7]
    num_truck = len(truck_detections)

    #Adding up the total number of detected vehicles 
    num_vehicle = num_bicycles + num_car + num_mcycle + num_bus + num_truck 
    return num_vehicle


image_path = "/home/lwcbwhite/TrafficAid/object_recognition/YOLOv8/car/bike.jpg" #image-path will be replaced with the webcam data
num_vehicle = counting_vehicles(image_path)
print("Fahrzeuge im Bild:", num_vehicle)

#Training and validation code not included

#The model was additionally trained over 3 epochs on the COCO dataset. Picture agumentation was tested,
#but not used, because the accuracy was already satisfying without any further training.
#If there are issues in a real life application, e.g with contrast, the model can be fine tuned with 
#alternated training data.

#To validate the model a seperate set of mosaic pictures was used to evalue the performance.
#Performance plots of recall, precision, F1 and confidence were created and used to evaluate the code.