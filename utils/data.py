import json
import shutil
import torch
import torchvision.transforms as T
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from random import sample, seed, shuffle
from collections import Counter


def read_txt(txt_path: str) -> list[list[float]]:
    """
    read annotation data from a single txt file
    """
    with open(txt_path, "r") as f:
        data = [i.strip() for i in f.readlines()]

    data = [[float(j) for j in i.split()] for i in data]
    return data


def read_json(json_file: str) -> dict:
    with open(json_file, "rb") as f:
        data = json.load(f)
    return data


def get_stats(files: list[str,],) -> pd.Series:
    """get class distribution statistics
    arguments:
        txts: list of paths to text files containing annotations
    returns:
        pd.Series with label as index and count as values
    """
    c = Counter()
    for json_file in files:
        temp = read_json(json_file)
        c.update(str(LABELS2ID.get(temp["chart-type"])))
            
    counts = pd.Series(dict(c.items())).sort_index()
    counts.index = counts.index.map(ID2LABELS)
    return counts


def read_image(img_file: str) -> Image:
    img = Image.open(img_file).convert("RGB")
    return img


# classes
ID2LABELS = {0: "vertical_bar",
             1: "horizontal_bar",
             2: "scatter",
             3: "dot",
             4: "line",
             5: "x_tick_label",
             6: "y_tick_label",
             7: "element",}

LABELS2ID = {"vertical_bar": 0,
             "horizontal_bar": 1,
             "scatter": 2,
             "dot": 3,
             "line": 4,
             "x_tick_label": 5,
             "y_tick_label": 6,
             "element": 7,}


def polygon2xyxy(polygon: dict) -> list:
    xs = [polygon["x0"], polygon["x1"], polygon["x2"], polygon["x3"]]
    ys = [polygon["y0"], polygon["y1"], polygon["y2"], polygon["y3"]]
    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
    return [xmin, ymin, xmax, ymax]


def bbox_dict2xyxy(bbox: dict) -> list:
    w = bbox["width"] 
    h = bbox["height"]
    xmin, ymin = bbox["x0"], bbox["y0"]
    xmax, ymax = xmin+w, ymin+h
    return [xmin, ymin, xmax, ymax]


def bbox_dict2vbar(bbox: dict) -> list:
    w = bbox["width"] 
    h = bbox["height"]
    xmin, ymin = bbox["x0"], bbox["y0"]
    xmax, ymax = xmin+w, ymin+h
    return [xmin, ymin, xmax, ymax]


def bbox_dict2hbar(bbox: dict) -> list:
    w = bbox["width"] 
    h = bbox["height"]
    xmin, ymin = bbox["x0"], bbox["y0"]
    xmax, ymax = xmin+w, ymin+h
    return [xmin, ymin, xmax, ymax]


def xywh2xyxy(box: list):
    x, y, w, h = box
    res = x-(w/2), y-(h/2), x+(w/2), y+(h/2)
    return res


def xyxy2xywh(box: list):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    res = x1+(w/2), y1+(h/2), w, h
    return res


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
        element_boxes = [bbox_dict2vbar(i) for i in elems]
    elif chart_type == "horizontal_bar":
        element_boxes = [bbox_dict2hbar(i) for i in elems]
    else:
        # use scatter val as width and height of point
        element_boxes = [xywh2xyxy([i["x"], i["y"], scatter_val, scatter_val]) for i in elems[0]]
    
    # axes
    axes = data["axes"]
    xaxes = [i["id"] for i in axes["x-axis"]["ticks"]]
    
    # text boxes
    text = [i for i in data["text"] if i["role"] in ["tick_label", "tick_grouping"]]
    text_boxes = [polygon2xyxy(i["polygon"]) for i in text]
    text_labels = []
    for i in text:
        if i["id"] in xaxes:
            text_labels.append(LABELS2ID.get("x_tick_label"))
        else:
            text_labels.append(LABELS2ID.get("y_tick_label"))
    
    # get bounding box for chart area
    plot_bbox = bbox_dict2xyxy(data["plot-bb"])
    
    # collate all boxes and labels
    boxes = [plot_bbox] + element_boxes + text_boxes
    labels = [LABELS2ID.get(chart_type)] + [LABELS2ID.get("element")]*len(element_boxes) + text_labels
    assert len(boxes) == len(labels), f"number of boxes: {len(boxes)} must equal number of labels: {len(labels)} in {json_file}!"
    return {"boxes": boxes, "labels": labels, "width": width, "height": height}


def validate_boxes(boxes: list) -> bool:
    for box in boxes:
        w = box[2] - box[0]
        h = box[3] - box[1]
        # width and height must be positive
        if w <= 0 or h <= 0:
            return False
    return True


def create_txt(annot_file: str) -> str:
    data = parse_json(annot_file)
    boxes = data["boxes"]
    if validate_boxes(boxes):
        labels = data["labels"]
        w, h = data["width"], data["height"]
        num_boxes = len(boxes)
        txt_data = ""
        for i in range(num_boxes):
            txt_data += f"{labels[i]} {boxes[i][0]/w} {boxes[i][1]/h} {boxes[i][2]/w} {boxes[i][3]/h}\n"
        return txt_data
    else:
        print(f"skipped {annot_file}, found non-positive width and height!")
        return None


def generate_dataset(annotations: str, images: str=None, save_dir: str="./dataset/",
                    val_size: float=0.15, test_size: float=0.15, 
                    random_state: int=42, mini: int=None) -> None:
    # remove directory if it exists
    if Path(save_dir).exists():
        shutil.rmtree(str(Path(save_dir)))

    # directories for each split
    Path(save_dir, "train/").mkdir(exist_ok=True, parents=True)
    Path(save_dir, "val/").mkdir(exist_ok=True, parents=True)
    Path(save_dir, "test/").mkdir(exist_ok=True, parents=True)
    
    annots = sorted([str(i) for i in Path(annotations).glob("*.json")])
        
    if not images:
        images = annotations.replace("annotations", "images")
    imgs = sorted([str(i) for i in Path(images).glob("*.jpg")])
    
    assert len(annots)==len(imgs), f"number of annotations: {len(annots)} not equal to {len(imgs)}!"
    
    if mini:  # use only a portion of the available data
        m = len(annots)
        seed(random_state)
        mini_indices = sample(range(m), mini)
        imgs = [imgs[i] for i in range(m) if i in mini_indices]
        annots = [annots[i] for i in range(m) if i in mini_indices]
    
    # split dataset
    num_rows = len(imgs)
    test_size = int(test_size * num_rows)
    val_size = int(val_size * (num_rows-test_size))
    train_size = num_rows - val_size - test_size

    seed(random_state)
    indices = list(range(num_rows))
    shuffle(indices)

    train_indices = indices[: train_size]
    val_indices = indices[train_size: train_size+val_size]
    test_indices = indices[train_size+val_size: ]

    # train
    print("train data")
    for idx in tqdm(train_indices):
        img_file = imgs[idx]
        annot_file = annots[idx]
        train_dir = Path(save_dir, "train")
        split_copyto(img_file, annot_file, save_dir=train_dir)
    
    # validation
    print("validation data")
    for idx in tqdm(val_indices):
        img_file = imgs[idx]
        annot_file = annots[idx]
        val_dir = Path(save_dir, "val")
        split_copyto(img_file, annot_file, save_dir=val_dir)

    # test
    print("test data")
    for idx in tqdm(test_indices):
        img_file = imgs[idx]
        annot_file = annots[idx]
        test_dir = Path(save_dir, "test")
        split_copyto(img_file, annot_file, save_dir=test_dir)


def split_copyto(img_file: str, annot_file: str, save_dir: str) -> None:
        # create txt
        txt_data = create_txt(annot_file)
        if txt_data:
            # copy image and annotations
            Path(save_dir, "images").mkdir(parents=True, exist_ok=True)
            Path(save_dir, "annotations").mkdir(parents=True, exist_ok=True)
            Path(save_dir, "labels").mkdir(parents=True, exist_ok=True)
            
            shutil.copyfile(src=img_file, dst=str(Path(save_dir, "images", Path(img_file).name)))
            shutil.copyfile(src=annot_file, dst=str(Path(save_dir, "annotations", Path(annot_file).name)))
            
            txt_file = str(Path(save_dir, "labels", Path(annot_file).stem+".txt"))
            with open(txt_file, "w+") as f:
                f.write(txt_data)


def collate_fn(batch):
    return tuple(zip(*batch))


def xyxyn2xyxy(box: list, width: int, height: int):
    return box[0]*width, box[1]*height, box[2]*width, box[3]*height


class BeneTech(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: str, transforms=None, im_size=(320, 320)):

        self.image_dir = Path(dataset_dir, "images")
        self.label_dir = Path(dataset_dir, "labels")

        self.images = sorted([i.name for i in self.image_dir.glob("*.jpg")])
        self.labels = sorted([i.name for i in self.label_dir.glob("*.txt")])

        self.transforms = transforms
        self.im_size = im_size

    def __getitem__(self, idx):
        img_path = Path(self.image_dir, self.images[idx])
        label_path = Path(self.label_dir, self.labels[idx])
        
        # load image
        img = Image.open(img_path).convert("RGB")
        label_data = read_txt(label_path)

        # parse data
        labels = [int(row[0]) for row in label_data]
        w, h = self.im_size
        boxes = [xyxyn2xyxy(row[1:5], width=w, height=h) for row in label_data]
        boxes = torch.as_tensor(boxes)

        # area
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img), target

        return img, target

    def __len__(self):
        return len(self.images)


def get_transform():
    transforms = [T.Resize((320, 320), antialias=True),
                  T.PILToTensor(),
                  T.ConvertImageDtype(torch.float)]
    return T.Compose(transforms)

