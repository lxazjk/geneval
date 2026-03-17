"""
Evaluate generated images using Mask2Former (or other object detector model)
"""

import argparse
import json
import os
import re
import sys
import time
import multiprocessing as mp
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import mmdet
from mmdet.apis import inference_detector, init_detector

import open_clip
from clip_benchmark.metrics import zeroshot_classification as zsc
zsc.tqdm = lambda it, *args, **kwargs: it

# Get directory path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagedir", type=str)
    parser.add_argument("--outfile", type=str, default="results.jsonl")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="./")
    # Other arguments
    parser.add_argument("--options", nargs="*", type=str, default=[])
    args = parser.parse_args()
    args.options = dict(opt.split("=", 1) for opt in args.options)
    if args.model_config is None:
        args.model_config = os.path.join(
            os.path.dirname(mmdet.__file__),
            "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
        )
    return args

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda"

def timed(fn):
    def wrapper(*args, **kwargs):
        startt = time.time()
        result = fn(*args, **kwargs)
        endt = time.time()
        print(f'Function {fn.__name__!r} executed in {endt - startt:.3f}s', file=sys.stderr)
        return result
    return wrapper

# Load models

@timed
def load_models(args):
    CONFIG_PATH = args.model_config
    OBJECT_DETECTOR = args.options.get('model', "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco")
    CKPT_PATH = os.path.join(args.model_path, f"{OBJECT_DETECTOR}.pth")
    object_detector = init_detector(CONFIG_PATH, CKPT_PATH, device=DEVICE)

    clip_arch = args.options.get('clip_model', "ViT-L-14")
    clip_model, _, transform = open_clip.create_model_and_transforms(clip_arch, pretrained="openai", device=DEVICE)
    tokenizer = open_clip.get_tokenizer(clip_arch)

    with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
        classnames = [line.strip() for line in cls_file]

    return object_detector, (clip_model, transform, tokenizer), classnames


COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
COLOR_CLASSIFIERS = {}

# Evaluation parts

class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects, transform, *, crop: bool, bgcolor: str):
        self._image = image.convert("RGB")
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects
        self._transform = transform
        self._crop = crop

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (index, self._image.size[::-1], mask.shape)
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        if self._crop:
            image = image.crop(box[:4])
        # if args.save:
        #     base_count = len(os.listdir(args.save))
        #     image.save(os.path.join(args.save, f"cropped_{base_count:05}.png"))
        return (self._transform(image), 0)


def color_classification(image, bboxes, classname):
    if classname not in COLOR_CLASSIFIERS:
        COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
            clip_model, tokenizer, COLORS,
            [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                f"a photo of a {{c}} object"
            ],
            DEVICE
        )
    clf = COLOR_CLASSIFIERS[classname]

    # 多卡/多进程场景下，DataLoader 再起 worker 很容易触发嵌套多进程/全局变量不可用问题。
    # 因此默认：多卡(num_gpus>1)时 clip_num_workers=0；单卡时维持原默认 4。
    crop = args.options.get('crop', '1') == '1'
    bgcolor = args.options.get('bgcolor', "#999")
    num_gpus = int(args.options.get('num_gpus', 1))
    clip_num_workers = int(args.options.get('clip_num_workers', 0 if num_gpus > 1 else 4))
    dataloader = torch.utils.data.DataLoader(
        ImageCrops(image, bboxes, transform, crop=crop, bgcolor=bgcolor),
        batch_size=16, num_workers=clip_num_workers
    )
    with torch.no_grad():
        pred, _ = zsc.run_classification(clip_model, clf, dataloader, DEVICE)
        return [COLORS[index.item()] for index in pred.argmax(1)]


def compute_iou(box_a, box_b):
    area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn([
        max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    ])
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def relative_position(obj_a, obj_b):
    """Give position of A relative to B, factoring in object dimensions"""
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    #
    revised_offset = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    #
    dx, dy = revised_offset / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5: relations.add("left of")
    if dx > 0.5: relations.add("right of")
    if dy < -0.5: relations.add("above")
    if dy > 0.5: relations.add("below")
    return relations


def evaluate(image, objects, metadata):
    """
    Evaluate given image using detected objects on the global metadata specifications.
    Assumptions:
    * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
    * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
    * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
        therefore, objects are expected to appear in sorted order
    """
    correct = True
    reason = []
    matched_groups = []
    # Check for expected objects
    for req in metadata.get('include', []):
        classname = req['class']
        matched = True
        found_objects = objects.get(classname, [])[:req['count']]
        if len(found_objects) < req['count']:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
        else:
            if 'color' in req:
                # Color check
                colors = color_classification(image, found_objects, classname)
                if colors.count(req['color']) < req['count']:
                    correct = matched = False
                    reason.append(
                        f"expected {req['color']} {classname}>={req['count']}, found " +
                        f"{colors.count(req['color'])} {req['color']}; and " +
                        ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                    )
            if 'position' in req and matched:
                # Relative position check
                expected_rel, target_group = req['position']
                if matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found_objects:
                        for target_obj in matched_groups[target_group]:
                            true_rels = relative_position(obj, target_obj)
                            if expected_rel not in true_rels:
                                correct = matched = False
                                reason.append(
                                    f"expected {classname} {expected_rel} target, found " +
                                    f"{' and '.join(true_rels)} target"
                                )
                                break
                        if not matched:
                            break
        if matched:
            matched_groups.append(found_objects)
        else:
            matched_groups.append(None)
    # Check for non-expected objects
    for req in metadata.get('exclude', []):
        classname = req['class']
        if len(objects.get(classname, [])) >= req['count']:
            correct = False
            reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
    return correct, "\n".join(reason)


def evaluate_image(filepath, metadata):
    result = inference_detector(object_detector, filepath)
    bbox = result[0] if isinstance(result, tuple) else result
    segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    image = ImageOps.exif_transpose(Image.open(filepath))
    detected = {}
    # Determine bounding boxes to keep
    confidence_threshold = THRESHOLD if metadata['tag'] != "counting" else COUNTING_THRESHOLD
    for index, classname in enumerate(classnames):
        ordering = np.argsort(bbox[index][:, 4])[::-1]
        ordering = ordering[bbox[index][ordering, 4] > confidence_threshold] # Threshold
        ordering = ordering[:MAX_OBJECTS].tolist() # Limit number of detected objects per class
        detected[classname] = []
        while ordering:
            max_obj = ordering.pop(0)
            detected[classname].append((bbox[index][max_obj], None if segm is None else segm[index][max_obj]))
            ordering = [
                obj for obj in ordering
                if NMS_THRESHOLD == 1 or compute_iou(bbox[index][max_obj], bbox[index][obj]) < NMS_THRESHOLD
            ]
        if not detected[classname]:
            del detected[classname]
    # Evaluate
    is_correct, reason = evaluate(image, detected, metadata)
    return {
        'filename': filepath,
        'tag': metadata['tag'],
        'prompt': metadata['prompt'],
        'correct': is_correct,
        'reason': reason,
        'metadata': json.dumps(metadata),
        'details': json.dumps({
            key: [box.tolist() for box, _ in value]
            for key, value in detected.items()
        })
    }


def collect_tasks(imagedir):
    tasks = []
    for subfolder in os.listdir(imagedir):
        folderpath = os.path.join(imagedir, subfolder)
        if not os.path.isdir(folderpath) or not subfolder.isdigit():
            continue
        with open(os.path.join(folderpath, "metadata.jsonl")) as fp:
            metadata = json.load(fp)
        samples_dir = os.path.join(folderpath, "samples")
        for imagename in os.listdir(samples_dir):
            imagepath = os.path.join(samples_dir, imagename)
            if not os.path.isfile(imagepath) or not re.match(r"\d+\.png", imagename):
                continue
            tasks.append((imagepath, metadata))
    return tasks


def _parse_gpu_ids(gpu_ids_str, num_gpus):
    if gpu_ids_str is None:
        return list(range(num_gpus))
    gpu_ids = [int(x) for x in gpu_ids_str.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("gpu_ids is empty")
    return gpu_ids


def _worker(gpu_id, tasks, args_in, constants, out_q):
    # IMPORTANT: set CUDA device before initializing models.
    torch.cuda.set_device(gpu_id)

    # Make globals available for ImageCrops / evaluate_image.
    global args, object_detector, clip_model, transform, tokenizer, classnames
    global THRESHOLD, COUNTING_THRESHOLD, MAX_OBJECTS, NMS_THRESHOLD, POSITION_THRESHOLD

    args = args_in
    THRESHOLD = constants["THRESHOLD"]
    COUNTING_THRESHOLD = constants["COUNTING_THRESHOLD"]
    MAX_OBJECTS = constants["MAX_OBJECTS"]
    NMS_THRESHOLD = constants["NMS_THRESHOLD"]
    POSITION_THRESHOLD = constants["POSITION_THRESHOLD"]

    object_detector, (clip_model, transform, tokenizer), classnames = load_models(args)
    local_results = []

    iterator = tasks
    if tqdm is not None:
        iterator = tqdm(tasks, desc=f"eval gpu{gpu_id}", dynamic_ncols=True)

    for imagepath, metadata in iterator:
        local_results.append(evaluate_image(imagepath, metadata))
    out_q.put(local_results)


def main(args):
    tasks = collect_tasks(args.imagedir)

    num_gpus = int(args.options.get('num_gpus', 1))
    gpu_ids = _parse_gpu_ids(args.options.get('gpu_ids', None), num_gpus)
    if num_gpus <= 1:
        iterator = tasks
        if tqdm is not None:
            iterator = tqdm(tasks, desc="eval", dynamic_ncols=True)
        full_results = [evaluate_image(imagepath, metadata) for imagepath, metadata in iterator]
    else:
        if len(gpu_ids) != num_gpus:
            raise ValueError(f"num_gpus={num_gpus} but gpu_ids has {len(gpu_ids)} entries")

        constants = {
            "THRESHOLD": THRESHOLD,
            "COUNTING_THRESHOLD": COUNTING_THRESHOLD,
            "MAX_OBJECTS": MAX_OBJECTS,
            "NMS_THRESHOLD": NMS_THRESHOLD,
            "POSITION_THRESHOLD": POSITION_THRESHOLD,
        }

        ctx = mp.get_context("spawn")
        out_q = ctx.Queue()
        procs = []
        chunks = [tasks[i::num_gpus] for i in range(num_gpus)]
        for rank, gpu_id in enumerate(gpu_ids):
            p = ctx.Process(target=_worker, args=(gpu_id, chunks[rank], args, constants, out_q))
            p.start()
            procs.append(p)

        full_results = []
        for _ in range(num_gpus):
            full_results.extend(out_q.get())

        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker process failed with exit code {p.exitcode}")
    # Save results
    if os.path.dirname(args.outfile):
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w") as fp:
        pd.DataFrame(full_results).to_json(fp, orient="records", lines=True)


if __name__ == "__main__":
    args = parse_args()
    THRESHOLD = float(args.options.get('threshold', 0.3))
    COUNTING_THRESHOLD = float(args.options.get('counting_threshold', 0.9))
    MAX_OBJECTS = int(args.options.get('max_objects', 16))
    NMS_THRESHOLD = float(args.options.get('max_overlap', 1.0))
    POSITION_THRESHOLD = float(args.options.get('position_threshold', 0.1))

    num_gpus = int(args.options.get('num_gpus', 1))
    if num_gpus <= 1:
        object_detector, (clip_model, transform, tokenizer), classnames = load_models(args)
    main(args)
