import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import json
import yaml
import os

from multiprocessing import Pool, Manager, Process
# from multiprocess import Pool, Manager
from itertools import repeat
from functools import partial
import copy

from tqdm import tqdm

def add_annotation_(img, annotations, shared_dict, counter, source):
    return_value = []
    file_name = img["file_name"]
    source_list_dir = os.listdir(source)
    if file_name not in source_list_dir:
        return
    # annotations = copy.deepcopy(annotations)
    # return_value = list(map(lambda annotation: annotation.pop("segmentation"); annotation if img["id"] == annotation["image_id"] else None, annotations))
    for annotation in annotations:
        if img["id"] == annotation["image_id"]:
            annotation.pop("segmentation")
            return_value.append(annotation)
    shared_dict.update({file_name: return_value})
    # print(f"{file_name} done")
    counter.value += 1
    print(f"{counter.value}/{len(source_list_dir)}: {counter.value / len(source_list_dir) * 100:.3f} done")

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        # modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        modelc = modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1


    t0 = time.time()

    # Load annotations
    annotation_source = opt.annotation
    print(annotation_source)
    
    if not os.path.exists(annotation_source):
        raise ValueError("Annotation directory does not exist")
    elif os.path.isfile(annotation_source): 

        # Parallel code
        manager = Manager()
        annotation_obj = manager.dict()
        annotation_file = annotation_source
        counter = manager.Value('i', 0)

        with open(annotation_file) as f:
            ann_obj = json.load(f)
            imgs = ann_obj["images"]
            annotations_global = ann_obj["annotations"]
        # pool = Pool(32)
        # add_annotation = partial(add_annotation_, annotations=annotations_global, shared_dict=annotation_obj, counter=counter, source=source)
        # pool.map_async(add_annotation, imgs)

        # pool.close()
        # pool.join()

        print("Process done")

    elif os.path.isdir(annotation_source):
        annotation_dir = annotation_source
        annotation_files = os.listdir(annotation_dir)
        annotation_obj = {}
        for annotation_file in annotation_files:
            if "json" not in annotation_file:
                continue
            with open(os.path.join(annotation_dir, annotation_file)) as f:
                annotations = json.load(f)
                annotation_obj[annotations["images"][0]["file_name"]] = annotations["annotations"]

    os.makedirs(opt.log_dir, exist_ok=True)

    

    result_log = {}
    for path, img, img_original, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0] 
            # zeroth value is the concatenation of the first value
            # The first value is the result of the various resolution of the images
        t2 = time_synchronized()

        # Prepare the target
        # original_shape = shapes[0]
        width_ratio = img_original.shape[1] / img.shape[3]
        height_ratio = img_original.shape[0] / img.shape[2]
        # width_ratio = original_shape[1] / img.shape[3]
        # height_ratio = original_shape[0] / img.shape[2]

        # target_annotations = annotation_obj[path.split("/")[-1]]
        # target_class_map = torch.zeros(img.shape[2], img.shape[3]).to(device)
        # for target in target_annotations:
        #     target_bbox = target["bbox"] # xywh
        #     target_bbox = [target_bbox[0] / width_ratio, target_bbox[1] / height_ratio, target_bbox[2] / width_ratio, target_bbox[3] / height_ratio]
        #     target_class_map[round(target_bbox[1]):round(target_bbox[1] + target_bbox[3]), round(target_bbox[0]):round(target_bbox[0] + target_bbox[2])] = 1

        # Caculate categorical distribution likelihood
        total_negative_log_likelihood_list = []
        for pred_elem in pred:
            pred_elem = pred_elem[0].to(torch.float32)
            pred_elem = torch.max(pred_elem, dim=0)[0]

            breakpoint()
            
            # print(target_class_map.shape, pred_elem.shape)
            if not target_class_map.shape[0] // pred_elem.shape[0] == target_class_map.shape[1] // pred_elem.shape[1]:
                # print(target_class_map.shape, pred_elem.shape)
                raise ValueError("The shape of the target and the prediction is not the same")
            # assert target_class_map.shape[0] // pred_elem.shape[0] == target_class_map.shape[1] // pred_elem.shape[1]
            assert target_class_map.shape[0] % pred_elem.shape[0] == 0
            shrink_ratio = target_class_map.shape[0] // pred_elem.shape[0]
            pool_kernel_size = (shrink_ratio, shrink_ratio)

            adaptive_target_class_map = torch.nn.functional.avg_pool2d(target_class_map.unsqueeze(0).unsqueeze(0), kernel_size=pool_kernel_size).squeeze(0).squeeze(0)

            # Confidence
            confidence = pred_elem[..., 4] + 1e-10

            # Normalize object probability
            obj_prob = pred_elem[..., 5:]
            obj_prob = obj_prob + 1e-10
            if obj_prob.shape[-1] > 1:
                obj_prob = obj_prob[..., 0]

            # Real object probability
            log_obj_class_prob = torch.log(obj_prob).squeeze(-1) + torch.log(confidence)
            log_no_obj_prob = torch.log(1 - confidence)

            # XY value
            upper_right_xy = pred_elem[..., :2] # XY
            center_xy = torch.floor(upper_right_xy / shrink_ratio)

            center_xy = center_xy.long().view(-1, 2)
            center_xy = center_xy.long()
            center_xy[:, 0] = torch.clamp(center_xy[:, 0], 0, adaptive_target_class_map.shape[1] - 1)
            center_xy[:, 1] = torch.clamp(center_xy[:, 1], 0, adaptive_target_class_map.shape[0] - 1)

                
            obj_existence_mask = torch.where(adaptive_target_class_map > 0, torch.tensor([1]).to(device), torch.tensor([0]).to(device))
            existence_prob = log_obj_class_prob * adaptive_target_class_map
            no_existence_prob = log_no_obj_prob * (1 - adaptive_target_class_map)
            
            total_likelihood = existence_prob + no_existence_prob
            effective_code_num = obj_existence_mask.sum().item()
            if opt.only_answer:
                # total_likelihood = existence_prob
                total_likelihood = total_likelihood * obj_existence_mask
                effective_code_num = obj_existence_mask.sum().item()
            else:
                # total_likelihood = existence_prob + no_existence_prob
                effective_code_num = existence_prob.shape.numel()

            total_negative_log_likelihood_list.append([-total_likelihood.sum().item(), effective_code_num])
        # Apply NMS
        result_log_key = path.split("/")[-1]
        print(result_log_key)
        result_log[result_log_key] = total_negative_log_likelihood_list

    with open(f"{opt.log_dir}/{opt.log_name}.yaml", "w") as f:
        yaml.dump(result_log, f)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    
    # Custom
    parser.add_argument('--log_dir', type=str, default='likelihood_log', help='Log directory')
    parser.add_argument('--log_name', type=str, default='likelihood', help='Log file name')
    parser.add_argument('--annotation', type=str, default='../real_datasets/train/labels', help='Annotation directory')
    parser.add_argument('--only-answer', action='store_true', help='Only answer the question')
    
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
