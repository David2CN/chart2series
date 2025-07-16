import json
import shutil
from pathlib import Path
from random import seed, sample

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm


# classes
ID2LABELS = {0: "vertical_bar",
             1: "horizontal_bar",
             2: "scatter",
             3: "dot",
             4: "line",
             5: "x_tick_label",
             6: "y_tick_label",
             7: "element"}

LABELS2ID = {"vertical_bar": 0,
             "horizontal_bar": 1,
             "scatter": 2,
             "dot": 3,
             "line": 4,
             "x_tick_label": 5,
             "y_tick_label": 6,
             "element": 7}


def read_image(img_file: str) -> Image:
    img = Image.open(img_file).convert("RGB")
    return img


def read_json(json_file: str) -> dict:
    with open(json_file, "rb") as f:
        data = json.load(f)
    return data


def polygon2xywhn(polygon: dict, width: float, height: float) -> list:
    xs = [polygon["x0"], polygon["x1"], polygon["x2"], polygon["x3"]]
    ys = [polygon["y0"], polygon["y1"], polygon["y2"], polygon["y3"]]
    w = (max(xs) - min(xs))
    h = (max(ys) - min(ys))
    x_center = (min(xs) + (w/2)) / width
    y_center = (min(ys) + (h/2)) / height
    return [x_center, y_center, w/width , h/height]


def bbox_dict2xywhn(bbox: dict, width: float, height: float) -> list:
    w = bbox["width"] 
    h = bbox["height"]
    x_center = (bbox["x0"] + (w/2)) / width
    y_center = (bbox["y0"] + (h/2)) / height
    return [x_center, y_center, w/width , h/height]


def bbox_dict2vbar(bbox: dict, width: float, height: float, scatter_val: float=15.0) -> list:
    w = bbox["width"] 
    h = bbox["height"]
    x_center = (bbox["x0"] + (w/2)) / width
    top = bbox["y0"] / height
    return [x_center, top, scatter_val/width, scatter_val/height]

def bbox_dict2hbar(bbox: dict, width: float, height: float, scatter_val: float=15.0) -> list:
    w = bbox["width"] 
    h = bbox["height"]
    right = (bbox["x0"] + w) / width
    y_center = (bbox["y0"] + (h/2)) / height
    return [right, y_center, scatter_val/width, scatter_val/height]


def xywhn2xyxy(box: list, width: float, height: float):
    x, y, w, h = box
    xmin, xmax = (x - (w/2))*width, (x + (w/2))*width
    ymin, ymax = (y - (h/2))*height, (y + (h/2))*height
    return [xmin, ymin, xmax, ymax]


def xywh2xyxy(box: list):
    x, y, w, h = box
    xmin, xmax = x - (w/2), x + (w/2)
    ymin, ymax = y - (h/2), y + (h/2)
    return [xmin, ymin, xmax, ymax]


def parse_json(json_file: str, 
               labels2id: dict,
               img_file: str=None, 
               scatter_val: float=15.0) -> dict:
    # read annotation
    data = read_json(json_file)
    
    # read image to get size
    if not img_file:
        img_file = json_file.replace("annotations", "images").replace(".json", ".jpg")
        
    img = read_image(img_file)
    width, height = img.size
    
    # get visual elements boxes
    elements = {"vertical_bar": "bars",
                 "horizontal_bar": "bars",
                 "scatter": "scatter points",
                 "line": "lines",
                 "dot": "dot points"}
    
    chart_type = data["chart-type"]
    elems = data["visual-elements"][elements.get(chart_type)]
    if chart_type == "vertical_bar":    
        element_boxes = [bbox_dict2vbar(i, width=width, height=height, scatter_val=scatter_val) for i in elems]
    elif chart_type == "horizontal_bar":
        element_boxes = [bbox_dict2hbar(i, width=width, height=height, scatter_val=scatter_val) for i in elems]
    else:
        # use scatter val as width and height of point
        element_boxes = [[i["x"]/width, i["y"]/height, scatter_val/width, scatter_val/height] for i in elems[0]]
    
    # axes
    axes = data["axes"]
    xaxes = [i["id"] for i in axes["x-axis"]["ticks"]]
    
    # text boxes
    text = [i for i in data["text"] if i["role"] in ["tick_label", "tick_grouping"]]
    text_boxes = [polygon2xywhn(i["polygon"], width=width, height=height) for i in text]
    text_labels = []
    for i in text:
        if i["id"] in xaxes:
            text_labels.append(labels2id.get("x_tick_label"))
        else:
            text_labels.append(labels2id.get("y_tick_label"))
    
    # get bounding box for chart area
    plot_bbox = bbox_dict2xywhn(data["plot-bb"], width=width, height=height)
    
    # collate all boxes and labels
    boxes = [plot_bbox] + element_boxes + text_boxes
    labels = [labels2id.get(chart_type)] + [labels2id.get("element")]*len(element_boxes) + text_labels
    assert len(boxes) == len(labels), f"number of boxes: {len(boxes)} must equal number of labels: {len(labels)} in {json_file}!"
    return {"boxes": boxes, "labels": labels}


def draw_bbox(img: Image, boxes: list, color: tuple=(0, 0, 255)) -> Image:
    width, height = img.size
    draw = ImageDraw.Draw(img)
    for bbox in boxes:
        draw.rectangle(xywhn2xyxy(bbox, width=width, height=height), outline=color)
    return img


def show_boxes(json_file: str, img_file: str=None, ax: plt.Axes=None):
    if img_file is None:
        img_file = json_file.replace("annotations", "images").replace(".json", ".jpg")

    img = read_image(img_file)
    d = parse_json(json_file)
    draw_img = draw_bbox(img, d["boxes"], color=(255, 0, 0))
    
    if not ax:
        plt.imshow(draw_img)
        plt.axis("off")
    else:
        ax.imshow(draw_img)
        ax.axis("off")


def polygon2xywhn(polygon: dict, width: float, height: float) -> list:
    xs = [polygon["x0"], polygon["x1"], polygon["x2"], polygon["x3"]]
    ys = [polygon["y0"], polygon["y1"], polygon["y2"], polygon["y3"]]
    w = (max(xs) - min(xs))
    h = (max(ys) - min(ys))
    x_center = (min(xs) + (w/2)) / width
    y_center = (min(ys) + (h/2)) / height
    return [x_center, y_center, w/width , h/height]


def bbox_dict2xywhn(bbox: dict, width: float, height: float) -> list:
    w = bbox["width"] 
    h = bbox["height"]
    x_center = (bbox["x0"] + (w/2)) / width
    y_center = (bbox["y0"] + (h/2)) / height
    return [x_center, y_center, w/width , h/height]


def bbox_dict2vbar(bbox: dict, width: float, height: float, scatter_val: float=15.0) -> list:
    w = bbox["width"] 
    h = bbox["height"]
    x_center = (bbox["x0"] + (w/2)) / width
    top = bbox["y0"] / height
    return [x_center, top, scatter_val/width, scatter_val/height]

def bbox_dict2hbar(bbox: dict, width: float, height: float, scatter_val: float=15.0) -> list:
    w = bbox["width"] 
    h = bbox["height"]
    right = (bbox["x0"] + w) / width
    y_center = (bbox["y0"] + (h/2)) / height
    return [right, y_center, scatter_val/width, scatter_val/height]


def xywhn2xyxy(box: list, width: float, height: float):
    x, y, w, h = box
    xmin, xmax = (x - (w/2))*width, (x + (w/2))*width
    ymin, ymax = (y - (h/2))*height, (y + (h/2))*height
    return [xmin, ymin, xmax, ymax]


def xywh2xyxy(box: list):
    x, y, w, h = box
    xmin, xmax = x - (w/2), x + (w/2)
    ymin, ymax = y - (h/2), y + (h/2)
    return [xmin, ymin, xmax, ymax]


def parse_json(json_file: str, img_file: str=None, scatter_val: float=15.0) -> dict:
    # read annotation
    data = read_json(json_file)
    
    # read image to get size
    if not img_file:
        img_file = json_file.replace("annotations", "images").replace(".json", ".jpg")
        
    img = read_image(img_file)
    width, height = img.size
    
    # get visual elements boxes
    elements = {"vertical_bar": "bars",
                 "horizontal_bar": "bars",
                 "scatter": "scatter points",
                 "line": "lines",
                 "dot": "dot points"}
    
    chart_type = data["chart-type"]
    elems = data["visual-elements"][elements.get(chart_type)]
    if chart_type == "vertical_bar":    
        element_boxes = [bbox_dict2vbar(i, width=width, height=height, scatter_val=scatter_val) for i in elems]
    elif chart_type == "horizontal_bar":
        element_boxes = [bbox_dict2hbar(i, width=width, height=height, scatter_val=scatter_val) for i in elems]
    else:
        # use scatter val as width and height of point
        element_boxes = [[i["x"]/width, i["y"]/height, scatter_val/width, scatter_val/height] for i in elems[0]]
    
    # axes
    axes = data["axes"]
    xaxes = [i["id"] for i in axes["x-axis"]["ticks"]]
    
    # text boxes
    text = [i for i in data["text"] if i["role"] in ["tick_label", "tick_grouping"]]
    text_boxes = [polygon2xywhn(i["polygon"], width=width, height=height) for i in text]
    text_labels = []
    for i in text:
        if i["id"] in xaxes:
            text_labels.append(LABELS2ID.get("x_tick_label"))
        else:
            text_labels.append(LABELS2ID.get("y_tick_label"))
    
    # get bounding box for chart area
    plot_bbox = bbox_dict2xywhn(data["plot-bb"], width=width, height=height)
    
    # collate all boxes and labels
    boxes = [plot_bbox] + element_boxes + text_boxes
    labels = [LABELS2ID.get(chart_type)] + [LABELS2ID.get("element")]*len(element_boxes) + text_labels
    assert len(boxes) == len(labels), f"number of boxes: {len(boxes)} must equal number of labels: {len(labels)} in {json_file}!"
    return {"boxes": boxes, "labels": labels}


def create_txt(annot_file: str) -> str:
    data = parse_json(annot_file)
    boxes = data["boxes"]
    labels = data["labels"]
    num_boxes = len(boxes)
    txt_data = ""
    for i in range(num_boxes):
        txt_data += f"{labels[i]} {boxes[i][0]} {boxes[i][1]} {boxes[i][2]} {boxes[i][3]}\n"
    return txt_data


def generate_dataset(annotations: str, images: str=None, save_dir: str="./dataset/",
                    val_split: float=0.2, random_state: int=42, mini: int=None):
    if Path(save_dir, "images/").exists():
        shutil.rmtree(str(Path(save_dir, "images/")))
    if Path(save_dir, "labels/").exists():
        shutil.rmtree(str(Path(save_dir, "labels/")))
    if Path(save_dir, "annotations/").exists():
        shutil.rmtree(str(Path(save_dir, "annotations/")))
        
    # images
    Path(save_dir, "images/train/").mkdir(exist_ok=True, parents=True)
    Path(save_dir, "images/val/").mkdir(exist_ok=True, parents=True)
    
    # annotations
    Path(save_dir, "annotations/train/").mkdir(exist_ok=True, parents=True)
    Path(save_dir, "annotations/val/").mkdir(exist_ok=True, parents=True)
    
    # yolo labels
    Path(save_dir, "labels/train/").mkdir(exist_ok=True, parents=True)
    Path(save_dir, "labels/val/").mkdir(exist_ok=True, parents=True)
    
    annots = sorted([str(i) for i in Path(annotations).glob("*.json")])
    annots_names = [i.stem for i in Path(annotations).glob("*.json")]
        
    if not images:
        images = annotations.replace("annotations", "images")
    imgs = sorted([str(i) for i in Path(images).glob("*.jpg")])
    
    assert len(annots)==len(imgs), f"number of annotations: {len(annots)} not equal to {len(imgs)}!"
    
    if mini:  # use only a portion of the available data
        m = len(annots)
        indices = sample(range(m), mini)
        imgs = [imgs[i] for i in range(m) if i in indices]
        annots = [annots[i] for i in range(m) if i in indices]
    
    # split dataset
    num_rows = len(imgs)
    val_size = int(val_split * num_rows)

    seed(random_state)
    val_indices = sample(range(num_rows), val_size)
    train_indices = [i for i in range(num_rows) if i not in val_indices]
    
    # train
    print("train data")
    for idx in tqdm(train_indices):
        img_file = imgs[idx]
        annot_file = annots[idx]
        
        # copy image and annotations
        shutil.copyfile(src=img_file, dst=str(Path(save_dir, "images/train/", Path(img_file).name)))
        shutil.copyfile(src=annot_file, dst=str(Path(save_dir, "annotations/train/", Path(annot_file).name)))
        
        # create txt
        txt_data = create_txt(annot_file)    
        txt_file = str(Path(save_dir, "labels/train/", Path(annot_file).stem+".txt"))
        with open(txt_file, "w+") as f:
            f.write(txt_data)
    
    # validation
    print("validation data")
    for idx in tqdm(val_indices):
        img_file = imgs[idx]
        annot_file = annots[idx]
        
        # copy image and annotations
        shutil.copyfile(src=img_file, dst=str(Path(save_dir, "images/val/", Path(img_file).name))) 
        shutil.copyfile(src=annot_file, dst=str(Path(save_dir, "annotations/val/", Path(annot_file).name)))
        
        # create txt
        txt_data = create_txt(annot_file)
        txt_file = str(Path(save_dir, "labels/val/", Path(annot_file).stem+".txt"))
        with open(txt_file, "w+") as f:
            f.write(txt_data)

