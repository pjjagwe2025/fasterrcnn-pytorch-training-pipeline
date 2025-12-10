"""
Run evaluation on a trained model to get mAP, Precision, Recall and class wise AP.

USAGE:
python eval.py --data data_configs/voc.yaml --weights outputs/training/fasterrcnn_convnext_small_voc_15e_noaug/best_model.pth --model fasterrcnn_convnext_small
"""
from datasets import (
    create_valid_dataset, create_valid_loader
)
from models.create_fasterrcnn_model import create_model
from torch_utils import utils
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
from tqdm import tqdm

import torch
import argparse
import yaml
import torchvision
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', 
        default='data_configs/test_image_config.yaml',
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', 
        default='fasterrcnn_resnet50_fpn_v2',
        help='name of the model'
    )
    parser.add_argument(
        '-mw', '--weights', 
        default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-ims', '--imgsz', 
        default=640, 
        type=int, 
        help='image size to feed to the network'
    )
    parser.add_argument(
        '-w', '--workers', default=4, type=int,
        help='number of workers for data processing/transforms/augmentations'
    )
    parser.add_argument(
        '-b', '--batch', 
        default=8, 
        type=int, 
        help='batch size to load the data'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='show class-wise mAP'
    )
    parser.add_argument(
        '-st', '--square-training',
        dest='square_training',
        action='store_true',
        help='Resize images to square shape instead of aspect ratio resizing \
              for single image training. For mosaic training, this resizes \
              single images to square shape first then puts them on a \
              square canvas.'
    )
    args = vars(parser.parse_args())

    # Load the data configurations
    with open(args['data']) as file:
        data_configs = yaml.safe_load(file)

    # Validation settings and constants.
    try: # Use test images if present.
        VALID_DIR_IMAGES = data_configs['TEST_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['TEST_DIR_LABELS']
    except: # Else use the validation images.
        VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']
    NUM_WORKERS = args['workers']
    DEVICE = args['device']
    BATCH_SIZE = args['batch']

    # Model configurations
    IMAGE_SIZE = args['imgsz']

    # Load the pretrained model
    create_model = create_model[args['model']]
    if args['weights'] is None:
        try:
            model, coco_model = create_model(num_classes=NUM_CLASSES, coco_model=True)
        except:
            model = create_model(num_classes=NUM_CLASSES, coco_model=True)
        if coco_model:
            COCO_91_CLASSES = data_configs['COCO_91_CLASSES']
            valid_dataset = create_valid_dataset(
                VALID_DIR_IMAGES, 
                VALID_DIR_LABELS, 
                IMAGE_SIZE, 
                COCO_91_CLASSES, 
                square_training=args['square_training']
            )

    # Load weights.
    if args['weights'] is not None:
        model = create_model(num_classes=NUM_CLASSES, coco_model=False)
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        
        # Fix for DistributedDataParallel models
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('module.') for key in state_dict.keys()):
            print("Detected DistributedDataParallel model, removing 'module.' prefix...")
            # Create new state dict without 'module.' prefix
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.' prefix (len('module.') = 7)
                else:
                    new_key = key
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        valid_dataset = create_valid_dataset(
            VALID_DIR_IMAGES, 
            VALID_DIR_LABELS, 
            IMAGE_SIZE, 
            CLASSES,
            square_training=args['square_training']
        )
    model.to(DEVICE).eval()
    
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)

    @torch.inference_mode()
    def evaluate(
        model, 
        data_loader, 
        device, 
        out_dir=None,
        classes=None,
        colors=None
    ):
        metric = MeanAveragePrecision(
            class_metrics=args['verbose'],
            iou_type="bbox",
            iou_thresholds=[0.5],
            max_detection_thresholds=[100, 300, 500],
            extended_summary=True
        )
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"

        target = []
        preds = []
        counter = 0
        for images, targets in tqdm(metric_logger.log_every(data_loader, 100, header), total=len(data_loader)):
            counter += 1
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            with torch.no_grad():
                outputs = model(images)

            #####################################
            for i in range(len(images)):
                true_dict = dict()
                preds_dict = dict()
                true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
                true_dict['labels'] = targets[i]['labels'].detach().cpu()
                preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
                preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
                preds.append(preds_dict)
                target.append(true_dict)
            #####################################
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        torch.set_num_threads(n_threads)
        metric.update(preds, target)
        metric_summary = metric.compute()
        return metric_summary

    stats = evaluate(
        model, 
        valid_loader, 
        device=DEVICE,
        classes=CLASSES,
    )

    print('\n')
    print('='*70)
    print('STANDARD COCO METRICS')
    print('='*70)
    pprint(stats)
    
    # Extract precision and recall per class
    print('\n')
    print('='*70)
    print('PRECISION AND RECALL METRICS')
    print('='*70)
    
    precision_per_class = []
    recall_per_class = []
    
    try:
        # Get the precision and recall tensors
        precision_tensor = stats['precision']  # Shape: [1, 101, K, 4, 3]
        recall_tensor = stats['recall']  # Shape: [1, K, 4, 3]
        
        print(f"\nDebug - Precision shape: {precision_tensor.shape}, Recall shape: {recall_tensor.shape}")
        
        # Check if we have per-class metrics
        if precision_tensor.shape[2] > 1:
            # We have per-class metrics
            num_classes = len(CLASSES)
            for class_idx in range(1, num_classes):  # Skip background at index 0
                # Precision: average over recall thresholds at IoU=0.5, area=all, maxDets=100
                class_precision = precision_tensor[0, :, class_idx, 0, 0]
                valid_precision = class_precision[class_precision > -1]
                
                if len(valid_precision) > 0:
                    precision_per_class.append(valid_precision.mean().item())
                else:
                    precision_per_class.append(0.0)
                
                # Recall: single value at IoU=0.5, area=all, maxDets=100
                class_recall = recall_tensor[0, class_idx, 0, 0]
                
                if class_recall > -1:
                    recall_per_class.append(class_recall.item())
                else:
                    recall_per_class.append(0.0)
        else:
            # Aggregate metrics only (class dimension is 1)
            overall_prec = precision_tensor[0, :, 0, 0, 0]  # [101] recall thresholds
            valid_prec = overall_prec[overall_prec > -1]
            
            overall_rec = recall_tensor[0, 0, 0, 0]  # Single value
            
            if len(valid_prec) > 0:
                avg_precision = valid_prec.mean().item()
            else:
                avg_precision = 0.0
            
            if overall_rec > -1:
                avg_recall = overall_rec.item()
            else:
                avg_recall = 0.0
            
            # Assign same value to all classes since we don't have per-class breakdown
            precision_per_class = [avg_precision] * (len(CLASSES) - 1)
            recall_per_class = [avg_recall] * (len(CLASSES) - 1)
            print(f"Note: Using aggregate metrics (no per-class breakdown available)")
    
    except Exception as e:
        print(f"Warning: Could not extract per-class precision/recall: {e}")
        # Fallback to zeros
        precision_per_class = [0.0] * (len(CLASSES) - 1)
        recall_per_class = [0.0] * (len(CLASSES) - 1)
    
    # Calculate overall precision and recall
    if len(precision_per_class) > 0:
        overall_precision = sum(precision_per_class) / len(precision_per_class)
        overall_recall = sum(recall_per_class) / len(recall_per_class)
    else:
        overall_precision = 0.0
        overall_recall = 0.0
    
    print(f"\nOverall Precision: {overall_precision:.3f}")   
    print(f"Overall Recall: {overall_recall:.3f}")
    
    if args['verbose']:
        print('\n')
        pprint(f"Classes: {CLASSES}")
        print('\n')
        print('AP / AR / Precision / Recall per class')
        empty_string = ''
        if len(CLASSES) > 2: 
            num_hyphens = 95
            print('-'*num_hyphens)
            print(f"|    | Class{empty_string:<16}| AP{empty_string:<10}| AR{empty_string:<10}| Precision{empty_string:<6}| Recall{empty_string:<9}|")
            print('-'*num_hyphens)
            class_counter = 0
            for i in range(0, len(CLASSES)-1, 1):
                class_counter += 1
                ap_val = np.array(stats['map_per_class'][i]).item() if i < len(stats['map_per_class']) else 0.0
                ar_val = np.array(stats['mar_100_per_class'][i]).item() if i < len(stats['mar_100_per_class']) else 0.0
                prec_val = precision_per_class[i] if i < len(precision_per_class) else 0.0
                rec_val = recall_per_class[i] if i < len(recall_per_class) else 0.0
                print(f"|{class_counter:<3} | {CLASSES[i+1]:<20} | {ap_val:.3f}{empty_string:<7}| {ar_val:.3f}{empty_string:<7}| {prec_val:.3f}{empty_string:<10}| {rec_val:.3f}{empty_string:<10}|")
            print('-'*num_hyphens)
            print(f"|Avg{empty_string:<23} | {np.array(stats['map']).item():.3f}{empty_string:<7}| {np.array(stats['mar_100']).item():.3f}{empty_string:<7}| {overall_precision:.3f}{empty_string:<10}| {overall_recall:.3f}{empty_string:<10}|")
        else:
            num_hyphens = 84
            print('-'*num_hyphens)
            print(f"|Class{empty_string:<10} | AP{empty_string:<10}| AR{empty_string:<10}| Precision{empty_string:<6}| Recall{empty_string:<9}|")
            print('-'*num_hyphens)
            ap_val = np.array(stats['map']).item()
            ar_val = np.array(stats['mar_100']).item()
            prec_val = precision_per_class[0] if len(precision_per_class) > 0 else 0.0
            rec_val = recall_per_class[0] if len(recall_per_class) > 0 else 0.0
            print(f"|{CLASSES[1]:<15} | {ap_val:.3f}{empty_string:<7}| {ar_val:.3f}{empty_string:<7}| {prec_val:.3f}{empty_string:<10}| {rec_val:.3f}{empty_string:<10}|")
            print('-'*num_hyphens)
            print(f"|Avg{empty_string:<12} | {ap_val:.3f}{empty_string:<7}| {ar_val:.3f}{empty_string:<7}| {overall_precision:.3f}{empty_string:<10}| {overall_recall:.3f}{empty_string:<10}|")
            print('-'*num_hyphens)
    
    print('\n')
    print('='*70)
    print('SUMMARY')
    print('='*70)
    print(f"mAP@0.5:0.95: {np.array(stats['map']).item():.3f}")
    print(f"mAP@0.5:     {np.array(stats['map_50']).item():.3f}")
    print(f"Precision:    {overall_precision:.3f}")
    print(f"Recall:       {overall_recall:.3f}")
    print('='*70)