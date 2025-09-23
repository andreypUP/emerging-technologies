from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11n-cls.pt")  
    
    # Train the model
    results = model.train(
        data="data/custom", # this is the parent directory of the dataset
        epochs=2,          # number of epochs for training
        imgsz=32,           # imgsz is also input_size
        workers=8           # in case of RunTimeError, reduce this value until you found enough workers
        )