import json
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog


class RegisterData:
    def __init__(self) -> None:
        
        pass    
    def read_json(self, ann_path):
        with open(ann_path, "r") as ann_file:
            ann_dicts = json.load(ann_file)
        
        return ann_dicts
    
    def register_dataset(self, dataset_name, ann_path, file_path):
        print(f"Dataset Registered!")
        register_coco_instances(dataset_name, {}, ann_path, file_path)

    def register_metadata(self, dataset_name, classes):
        MetadataCatalog.get(dataset_name).set(thing_classes=classes)
        metadata = MetadataCatalog.get(dataset_name)
        return metadata


