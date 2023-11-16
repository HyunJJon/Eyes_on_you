import torch

repo = "deepcam-cn/yolov5-face"


class YOLOv5FaceModel:
    def __init__(self, model_name="yolov5s"):
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        """Loads the model using torch.hub."""
        print(f"Loading model {self.model_name}...")
        # URL to the repository and the model

        model = torch.hub.load(
            repo,
            self.model_name,
            pretrained=True,
            source="github",
            force_reload=True,
        )
        print(f"Model {self.model_name} loaded.")
        return model


# Create a global model instance that can be imported and used in other modules
model = YOLOv5FaceModel().model

if __name__ == "__main__":
    # Test the model
    image = "https://ultralytics.com/images/zidane.jpg"
    results = model(image)
    results.show()
