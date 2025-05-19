import pickle
import base64
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import json
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import numpy as np
import pandas as pd

def convert_preds_pickle_to_coco(file, 
                                save_path= "/network/scratch/t/tengmeli/RSPrompter_clean/rspromter_anchor_trees_preds/predictions_coco_format.json"):
    with open(file, "rb") as f:
        data = pickle.load(f)
    
    coco_predictions = []

    for pred in data:  # Adjust based on your pickle structure
        for i in range(pred['pred_instances']["labels"].shape[0]):  # Replace 'objects' with the correct key
            mask = pred['pred_instances']['masks'][i]
            mask["counts"] = mask["counts"].decode("utf-8")
            coco_predictions.append({
                "image_id": pred['img_id'],  # Image ID
                "category_id": (pred['pred_instances']['labels'][i] + 1).item(),  # Category ID
                "segmentation": mask,  # Segmentation in RLE or polygon
                "bbox": pred['pred_instances']['bboxes'][i].tolist(),  # Bounding box [x, y, width, height]
                "score": pred['pred_instances']['scores'][i].item(),  # Confidence score
            })

    with open(save_path, "w") as f:
        json.dump(coco_predictions, f)


    
def main(preds_json= "/network/scratch/t/tengmeli/RSPrompter_clean/rspromter_anchor_trees_preds/predictions_coco_format.json"):
    print(f"evaluation of {preds_json}")
    annots =  "/network/projects/trees-co2/quebec_trees_tiles_fullresolution/merged_annots_test_new.json" 
    #"/network/projects/trees-co2/final_tiles/merged_annots_test_new.json"
    
    with open(preds_json, "r") as f:
        coco_preds = json.load(f)
        
    img_id_to_preds = {}
    for i, pred in enumerate(coco_preds):
        img_id = pred["image_id"]
        if img_id not in img_id_to_preds:
            img_id_to_preds[img_id] = [i]
        else:
            img_id_to_preds[img_id] += [i]
    # Load COCO ground truth
    coco_gt = COCO(annots)
    
    iou_list = []
    metric = MeanAveragePrecision(iou_type= "segm", class_metrics=True)
    metric_single = MeanAveragePrecision(iou_type="segm", class_metrics = True)
    
    preds_classes = []
    tar_classes = []
    
    for im_id in coco_gt.getImgIds():
        targets = []
        targets_single = []
        ann_ids = coco_gt.getAnnIds(imgIds=im_id)
        anns = coco_gt.loadAnns(ann_ids)

        # Extract boxes and labels
        boxes = []
        labels = []
        masks = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y,  w,  h])  # Convert to [x_min, y_min, x_max, y_max]
            labels.append(ann['category_id'])
            if 'counts' in ann["segmentation"] and isinstance(ann["segmentation"]['counts'], str):
                counts = base64.b64decode(ann["segmentation"]['counts'])
                ann["segmentation"]['counts'] = counts

            masks.append(maskUtils.decode(ann["segmentation"]))
        labels= np.array(labels)
        masks = np.array(masks)
        targets.append({"boxes": boxes, "labels": torch.Tensor(labels).type(torch.int),"masks": torch.Tensor(masks).type(torch.uint8)})
        targets_single.append({"boxes": boxes, "labels": torch.full_like(torch.Tensor(labels), 0).type(torch.int),"masks": torch.Tensor(masks).type(torch.uint8)})
        
        dict_pred = {"boxes": [], "scores": [], "labels": [], "masks":[]}
        dict_single = {"boxes": [], "scores": [], "labels": [], "masks":[]}
        
        if im_id in img_id_to_preds:
            for pred_idx in img_id_to_preds[im_id]:

                pred = coco_preds[pred_idx]
                x, y, w, h = pred['bbox']
                dict_pred["boxes"].append([x, y, x + w, y + h])
                dict_pred["scores"].append(pred["score"])
                dict_pred["labels"].append(pred["category_id"])

                dict_pred["masks"].append(maskUtils.decode(pred["segmentation"]))
            dict_pred["labels"] = torch.Tensor(dict_pred["labels"]).type(torch.int)
            dict_pred["masks"] = torch.Tensor(np.array(dict_pred["masks"])).type(torch.uint8)
            dict_pred["scores"] = torch.Tensor(dict_pred["scores"])
            
            dict_single["masks"]= dict_pred["masks"] 
            dict_single["scores"]= dict_pred["scores"] 
            dict_single["boxes"]= dict_pred["boxes"] 
            dict_single["labels"]= torch.full_like( dict_pred["labels"] , 0).type(torch.int)
        else:
            #empty predictions for this image
            dict_pred = {"boxes": [], "scores": torch.Tensor([]), "labels": torch.Tensor([]), "masks":torch.Tensor([])}
            dict_single = {"boxes": [], "scores": torch.Tensor([]), "labels": torch.Tensor([]), "masks":torch.Tensor([])}
        
        metric.update([dict_pred], targets)
        metric_single.update([dict_single], targets_single)
        
        preds = [dict_pred]
        iou_per_instance = []
        for k in range(len(targets)):
            for idxt, t in enumerate(targets[k]["masks"]):
                pred_class = -1
                max_iou = 0
                tar_class = targets[k]["labels"][idxt]
    
                for idx, p in enumerate(preds[k]["masks"]):
                    intersection = (t & p).sum().float()
                    union = (t | p).sum().float()
                    iou = float(intersection / union) if union > 0 else 0.0
                    if iou > max_iou:
                        max_iou=iou
                        pred_class = preds[k]["labels"][idx]
                iou_per_instance.append(max_iou) 
                tar_classes.append(tar_class.item())
                if pred_class == -1:
                    preds_classes.append(pred_class)
                else:   
                    preds_classes.append(pred_class.item())
        
                    
        iou_list.append(np.mean(iou_per_instance))
            
    a = metric.compute()
    print(a)
    b = metric_single.compute()
  
    print(b)    
    print("Iou", np.mean(iou_list))
    
    df = pd.DataFrame(
    {'preds': preds_classes,
     'targets': tar_classes
    })
    print("saving df")
    df.to_csv("/network/scratch/t/tengmeli/rsprompter_dsm_preds_tars_classes.csv")
    
    #metric_weighed = 0
    #weights = [1471,1056,544,6519,1946,1050,1601,19,56]
    #for k, elem in enumerate(a['map_per_class']):
    #    metric_weighed += elem * weights[k]
    #metric_weighed = metric_weighed/ np.sum(np.array(weights))
    #print("weighed map ", metric_weighed)
        
    """
    # Parse ground truth for TorchMetrics
    targets = []
    for img_id in coco_gt.getImgIds():
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)

        # Extract boxes and labels
        boxes = []
        labels = []
        masks = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]
            labels.append(ann['category_id'])
            if 'counts' in ann["segmentation"] and isinstance(ann["segmentation"]['counts'], str):
                counts = base64.b64decode(ann["segmentation"]['counts'])
                ann["segmentation"]['counts'] = counts

            masks.append(maskUtils.decode(ann["segmentation"]))
        labels= np.array(labels)
        masks = np.array(masks)
        targets.append({"boxes": boxes, "labels": torch.Tensor(labels),"masks": torch.Tensor(masks)})
    
    with open("/network/scratch/t/tengmeli/RSPrompter_clean/rspromter_anchor_trees_preds/predictions_coco_format.json", "r") as f:
        coco_preds = json.load(f)

    # Parse predictions for TorchMetrics
    preds = []
    img_id_to_preds = {}
    for pred in coco_preds:
        img_id = pred["image_id"]
        if img_id not in img_id_to_preds:
            img_id_to_preds[img_id] = {"boxes": [], "scores": [], "labels": [], "masks":[]}

        # Append prediction details
        x, y, w, h = pred['bbox']
        img_id_to_preds[img_id]["boxes"].append([x, y, x + w, y + h])
        img_id_to_preds[img_id]["scores"].append(pred["score"])
        img_id_to_preds[img_id]["labels"].append(pred["category_id"])
        img_id_to_preds[img_id]["masks"].append(torch.Tensor(maskUtils.decode(pred["segmentation"])))
    # Convert to list format
    img_id_to_preds[img_id]["labels"] = torch.Tensor(img_id_to_preds[img_id]["labels"])
    img_id_to_preds[img_id]["masks"] = torch.Tensor(img_id_to_preds[img_id]["masks"])
    img_id_to_preds[img_id]["scores"] = torch.Tensor(img_id_to_preds[img_id]["scores"])
    for img_id in coco_gt.getImgIds():
        preds.append(img_id_to_preds.get(img_id, {"boxes": [], "scores": [], "labels": [], "masks":[]}))

    metric = MeanAveragePrecision(iou_type= "segm")
    print("evaluation")
    #import pdb; pdb.set_trace()
    for i, elem in enumerate(preds):
        metric.update([preds[i]], [targets[i]])
    result = metric.compute()
    print(result)

    """    
if __name__=="__main__":
    save_path =  "/network/scratch/t/tengmeli/RSPrompter_SBL-dsm/sbl/preds_14_seed0_test.json" #"/network/scratch/t/tengmeli/RSPrompter_final/rsprompter-anchor-trees-base-dsm-final/predictions_coco_format_seed1337_best.json"
    file = "/network/scratch/t/tengmeli/RSPrompter_SBL-dsm/sbl/preds_14_seed0_test.pkl"
    #"/network/scratch/t/tengmeli/RSPrompter_final/rsprompter-anchor-trees-base-dsm-final/preds_seed1337_epoch_9.pkl"
    
    convert_preds_pickle_to_coco(file,save_path)
    main(save_path)
  
    """
    print("seed 42 best")
    save_path = "/network/scratch/t/tengmeli/RSPrompter_final/rsprompter-anchor-trees-base-dsm-final/predictions_coco_format_seed4021_best_dsm.json"
    file ="/network/scratch/t/tengmeli/RSPrompter_clean/rspromter_anchor_trees_new_preds/preds_seed4021.pkl" #"/network/scratch/t/tengmeli/RSPrompter_final/rsprompter-anchor-trees-base-dsm-final/preds_seed42_epoch_4.pkl"
    convert_preds_pickle_to_coco(file,save_path)
    main(save_path)
    
    
    print("seed 0 best")
    save_path = "/network/scratch/t/tengmeli/RSPrompter_final/rsprompter-anchor-trees-base-dsm-final/predictions_coco_format_seed3999_best_dsm.json"
    file = "/network/scratch/t/tengmeli/RSPrompter_clean/rspromter_anchor_trees_new_preds/preds_seed3999_best.pkl" 
    #"/network/scratch/t/tengmeli/RSPrompter_final/rsprompter-anchor-trees-base-dsm-final/preds_seed0_epoch_8.pkl"
    convert_preds_pickle_to_coco(file,save_path)
    main(save_path)
    
    print("seed 1337 best")
    save_path = "/network/scratch/t/tengmeli/RSPrompter_final/rsprompter-anchor-trees-base-dsm-final/predictions_coco_format_seed2040_best_dsm.json"
    file ="/network/scratch/t/tengmeli/RSPrompter_clean/rspromter_anchor_trees_new_preds/preds_seed2040.pkl" 
    #"/network/scratch/t/tengmeli/RSPrompter_final/rsprompter-anchor-trees-base-dsm-final/preds_seed1337_epoch_9.pkl"
    convert_preds_pickle_to_coco(file,save_path)
    main(save_path)
    
    #print("seed 42 last")
    #save_path = "/network/scratch/t/tengmeli/RSPrompter_clean/rspromter_anchor_trees_preds/predictions_coco_format_seed42.json"
    #file = "/network/scratch/t/tengmeli/RSPrompter_clean/rspromter_anchor_trees_preds/preds_seed42.pkl"
    #convert_preds_pickle_to_coco(file,save_path)
    #main(save_path)
    

    
    
    #print("seed 0 epoch 40")
    #main("/network/scratch/t/tengmeli/RSPrompter_clean/rspromter_anchor_trees_preds/predictions_coco_format.json")
    """
