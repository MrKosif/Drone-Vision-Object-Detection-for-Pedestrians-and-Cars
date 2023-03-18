from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

class Evaluate:

    def __init__(self) -> None:
        pass

    def coco_evaluation(self, dataset_name, cfg, predictor):
        evaluator = COCOEvaluator(dataset_name=dataset_name, distributed=False, output_dir="./output")
        val_loader = build_detection_test_loader(cfg, dataset_name="valid")
        inference_on_dataset(model=predictor.model, data_loader=val_loader, evaluator=evaluator)