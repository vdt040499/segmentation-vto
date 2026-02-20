"""Lightweight wrapper around a YOLO human-parsing checkpoint.

This module isolates model discovery/loading and exposes helper methods that
return reusable masks/crops for downstream virtual try-on stages.
"""

import sys
import os

import cv2
import numpy as np

from ultralytics import YOLO

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

class Human_Parsing():
    """Singleton wrapper that keeps one YOLO model instance per process."""

    __instance__ = None  

    @staticmethod
    def getInstance():
        """ Static access method to get the singleton instance. """
        if Human_Parsing.__instance__ is None:
            Human_Parsing()
        return Human_Parsing.__instance__
        
    def __init__(self):
        if Human_Parsing.__instance__ is not None:
            raise Exception("This class is a singleton! Use the getInstance() method to access the instance.")
        else:
            Human_Parsing.__instance__ = self  # Assign the instance to the class variable

            # Load pretrained model - check multiple locations so the same build
            # works locally, inside Docker, or on Hugging Face Spaces.
            root_model_path = os.path.join(os.getcwd(), "human_parsing_11l.pt")
            default_model_path = os.path.join(os.getcwd(), "./models/human_parsing", "human_parsing_11l.pt")
            
            # Initialize model as None - will be loaded when needed
            self.human_parsing_model = None
            self.model_path = None
            self.result_template = {}
            
            # Try to find and load default model if it exists
            if os.path.exists(root_model_path):
                self.model_path = root_model_path
                try:
                    self.human_parsing_model = YOLO(self.model_path, task = "segment")
                    # Each class keeps its own accumulator so downstream merges
                    # can reuse the same structure as in app.py.
                    self.result_template = {key: [] for key in self.human_parsing_model.names.values()}
                    print(f"MODEL LOADED: {self.model_path}")
                except Exception as e:
                    print(f"WARNING: Could not load model from {self.model_path}: {e}")
                    self.model_path = None
                    self.human_parsing_model = None
            elif os.path.exists(default_model_path):
                self.model_path = default_model_path
                try:
                    self.human_parsing_model = YOLO(self.model_path, task = "segment")
                    self.result_template = {key: [] for key in self.human_parsing_model.names.values()}
                    print(f"MODEL LOADED: {self.model_path}")
                except Exception as e:
                    print(f"WARNING: Could not load model from {self.model_path}: {e}")
                    self.model_path = None
                    self.human_parsing_model = None
            else:
                print("WARNING: No default model found. Please upload a model file.")
    
    def detect_cloth(self, frame, iou = 0.7, conf = 0.3):
        """Return a binary clothing mask, ignoring arm/leg classes."""
        if self.human_parsing_model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        result = self.human_parsing_model.predict(frame, iou = iou, conf = conf, verbose = False)[0].cpu()

        boxes = result.boxes

        final_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype = np.uint8)
        
        # Process result list
        if len(boxes) == 0:
            return final_mask
        
        boxes = boxes.data.tolist()
        masks = result.masks

        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype = np.uint8)

        for i in range(len(boxes)):
            pred_xmin, pred_ymin, pred_xmax, pred_ymax, lp_score, class_id = boxes[i]

            if self.human_parsing_model.names[int(class_id)] in ["arm", "leg"]:
                continue

            xy = masks[i].xy[0].astype(int)
            
            if len(xy) != 0:
                cv2.fillPoly(mask, [xy], color = 255)

        return mask
    
    def detect_category(self, frame, iou = 0.7, conf = 0.3):
        """Return YOLO label text plus per-category crops/masks."""
        if self.human_parsing_model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        result = self.human_parsing_model.predict(frame, iou = iou, conf = conf, verbose = False)[0].cpu()

        labels = ""  # YOLO label file string, reused by collaborators that need polygons

        # Pre-allocate slot for each body region so downstream consumers always
        # receive the same schema even when a class is absent in the frame.
        template_result = {
            "lowerbody":{
                "image": None,
                "box": None,
                "mask": np.zeros((frame.shape[0], frame.shape[1]), dtype = np.uint8),
            },
            "upperbody":{
                "image": None,
                "box": None,
                "mask": np.zeros((frame.shape[0], frame.shape[1]), dtype = np.uint8),
            },
            "wholebody":{
                "image": None,
                "box": None,
                "mask": np.zeros((frame.shape[0], frame.shape[1]), dtype = np.uint8),
            }
        }

        boxes = result.boxes

        if len(boxes) == 0:
            return labels, template_result
        
        boxes = boxes.data.tolist()
        masks = result.masks

        for i in range(len(boxes)):
            pred_xmin, pred_ymin, pred_xmax, pred_ymax, lp_score, class_id = boxes[i]
            xy = masks[i].xy[0].astype(int)

            class_name = self.human_parsing_model.names[int(class_id)]
            if class_name in ["lowerbody", "upperbody", "wholebody"]:
                if template_result[class_name]["box"] is None:
                    template_result[class_name]["box"] = [int(pred_xmin), int(pred_ymin), int(pred_xmax), int(pred_ymax)]
                else:
                    template_result[class_name]["box"] = [min(int(pred_xmin), template_result[class_name]["box"][0]), 
                                                          min(int(pred_ymin), template_result[class_name]["box"][1]),
                                                          max(int(pred_xmax), template_result[class_name]["box"][2]), 
                                                          max(int(pred_ymax), template_result[class_name]["box"][3])]
                    
                if len(xy) != 0:
                    cv2.fillPoly(template_result[class_name]["mask"], [xy], color = 1)
                    
            labels += f"{int(class_id)}"
            for (x, y)in xy.tolist(): 
                labels = labels + f" {x} {y}"
            labels += "\n"

        area_threshold = 0.1  # drop noisy detections that cover <10% of the frame
        frame_area = frame.shape[0] * frame.shape[1]

        for key in template_result.keys():
            if template_result[key]["box"] is not None:
                xmin, ymin, xmax, ymax = template_result[key]["box"]
                
                this_area = abs((xmax - xmin) * (ymax - ymin))

                if (this_area / frame_area) < area_threshold:
                    template_result[key]["box"] = None
                    continue

                img = np.copy(frame)

                # Fill image
                # masked_img = cv2.bitwise_and(img, img, mask=template_result[key]["mask"])
                # xmin, ymin, xmax, ymax = template_result[key]["box"]
                # cropped_img = masked_img[ymin:ymax, xmin:xmax]

                cropped_img = img[ymin:ymax, xmin:xmax]  # Keep raw RGB patches for virtual try-on preview

                template_result[key]["image"] = cropped_img

        return labels, template_result
    
if __name__ == "__main__":
    from imread_from_url import imread_from_url

    human_parsing = Human_Parsing.getInstance()

    # file_path = "https://media.thereformation.com/image/upload/f_auto,q_auto:eco,dpr_auto/w_500/PRD-SFCC/1315925/CUSTARD/1315925.2.CUSTARD?_s=RAABAB0"
    # img = imread_from_url(file_path)
    img = cv2.imread("dataset_shein/images/co52nkrb2hjb11ais0hg/co52nkrb2hjb11ais0hg_1.jpg")

    # Detect clothes only
    labels, result = human_parsing.detect_category(img)

    for key in result.keys():
        if result[key]["box"] is not None:
            cv2.imwrite(f"{key}.jpg", result[key]["image"])
