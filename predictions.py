import matplotlib.pyplot as plt
import cv2
from detectron2.utils.visualizer import Visualizer

class Predict:

    def __init__(self) -> None:
        pass

    def predict(self, image_path, predictor):
        img = cv2.imread(image_path)
        output = predictor(img)
        return output
        

    def visualize_prediction(self, image_path, bbox, metadata):
        img = cv2.imread(image_path)
        plt.figure(figsize=(16,10))

        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1)
        img = cv2.imread(image_path)
        visualizer = visualizer.draw_instance_predictions(bbox["instances"].to("cpu"))
        plt.imshow(visualizer.get_image()[:, :, ::-1])
