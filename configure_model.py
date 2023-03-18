from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import os

class ConfigurModel:
    def __init__(self) -> None:
        self.train_cfg = get_cfg()
        self.predict_cfg = get_cfg()

    def create_trainer(self, train_set, test_set, class_count, iteration, device):
        self.train_cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        self.train_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        self.train_cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
        self.train_cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
        self.train_cfg.INPUT.MIN_SIZE_TRAIN = 0
        self.train_cfg.INPUT.MAX_SIZE_TRAIN = 99999
        self.train_cfg.DATASETS.TRAIN = (train_set,)
        self.train_cfg.DATASETS.TEST = (test_set,)
        self.train_cfg.DATALOADER.NUM_WORKERS = 2
        self.train_cfg.SOLVER.IMS_PER_BATCH = 2
        self.train_cfg.SOLVER.BASE_LR = 0.00125
        self.train_cfg.SOLVER.MAX_ITER = iteration
        self.train_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.train_cfg.MODEL.ROI_HEADS.NUM_CLASSES = class_count
        self.train_cfg.MODEL.DEVICE = device

        os.makedirs(self.train_cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.train_cfg)
        trainer.resume_or_load(resume=False)
        return trainer
    
    def create_predictor(self, model_path, treshold, class_count, device="gpu"):
        self.predict_cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        self.predict_cfg.MODEL.WEIGHTS = model_path
        self.predict_cfg.SOLVER.IMS_PER_BATCH = 2
        self.predict_cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
        self.predict_cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
        self.predict_cfg.MODEL.ROI_HEADS.NUM_CLASSES = class_count
        self.predict_cfg.SOLVER.IMS_PER_BATCH = 2
        self.predict_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.predict_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = treshold
        self.predict_cfg.INPUT.MIN_SIZE_TEST = 0
        self.predict_cfg.INPUT.MAX_SIZE_TEST = 99999
        self.predict_cfg.MODEL.DEVICE = device
        predictor = DefaultPredictor(self.predict_cfg)
        return predictor