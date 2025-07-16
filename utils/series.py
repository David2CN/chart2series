from io import BytesIO

import numpy as np
from PIL import Image
from easyocr import Reader
# from pytesseract import pytesseract


def collapse_y_axis_boxes(ticks: list):
    if len(ticks) < 2:
        return ticks
    
    boxes = sorted(ticks, key=lambda x: x[1])
    y1, y2 = boxes[0], boxes[-1]   # first and last boxes  
    
    # height of combined box
    h = (y2[1] - y1[1]) + (y1[3]/2) + (y2[3]/2) + 0.02
    
    # width of combined box
    w = max([i[2] for i in boxes]) + 0.01
    
    # centers
    x = np.mean([i[0] for i in boxes])  # mean value of centers
    # ycenter of first box + half the full height
    # subtract halt the height of the first box and a tolerance
    y = (y1[1]) + (h/2) - (y1[3]/2) - 0.01
    
    collapsed = [x, y, w, h]
    return collapsed


def collapse_x_axis_boxes(ticks: list):
    if len(ticks) < 2:
        return ticks
    
    boxes = sorted(ticks, key=lambda x: x[0])
    x1, x2 = boxes[0], boxes[-1]   # first and last boxes  
    
    # width of combined box
    w = (x2[0] - x1[0]) + (x1[2]/2) + (x2[2]/2) + 0.02
    
    # width of combined box
    h = max([i[3] for i in boxes]) + 0.01
    
    # centers
    y = np.mean([i[1] for i in boxes])  # mean value of centers
    x = (x1[0]) + (w/2) - (x1[3]/2) -0.01
    
    collapsed = [x, y, w, h]
    return collapsed


def crop_image(crop: list, img: Image) -> Image:
    w, h = img.size
    xyxy_crop = xywhn2xyxy(crop, width=w, height=h)
    cropped_img = img.crop(xyxy_crop) #.resize((int(scale*w), int(scale*h)))
    b = BytesIO()
    cropped_img.save(b, format="jpeg")
    cropped_img = Image.open(b)
    return cropped_img


def expand_box(box: list):
    x, y, w, h = box
    return [x, y, w*1.1, h*1.1]


def to_polygon(points: list):
    polygon = {
        "x0": points[0][0],
        "x1": points[1][0],
        "x2": points[2][0],
        "x3": points[3][0],
        "y0": points[0][1],
        "y1": points[1][1],
        "y2": points[2][1],
        "y3": points[3][1],
    }
    return polygon


def words_crop_to_text(crop: list, img: Image, reader: Reader, axis: int=0):
    cropped_img = crop_image(crop, img)
    text = reader.readtext(cropped_img)
    if text == []:
        return 'aeiou'
    else:
        return text[0][1]
    

def nums_crop_to_text(crop: list, img: Image, c_num: int=6):
    cropped_img = crop_image(crop, img)
    text = pytesseract.image_to_string(cropped_img, lang='eng',
                                       config=f'--psm {c_num} -c tessedit_char_whitelist=01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.')
    text = text.replace("\f","").strip()
    try:
        text_val = float(text)
    except:
        text_val = None
    return text_val


def parse_boxes(boxes: list, labels: list):
    """ parse boxes and return a dictionary of which box is what
    """
    m = len(boxes)
    res = {"chart_box": [0.15, 0.2, 0.79, 0.75],
           "chart_type": 0,
           "elements": [],
           "x_ticks": [],
           "y_ticks": []
          }
    for i in range(m):
        if labels[i] < 5:
            res["chart_type"] = labels[i]
            res["chart_box"] = boxes[i]
        elif labels[i] == 5:
            res["x_ticks"].append(boxes[i])
        elif labels[i] == 6:
            res["y_ticks"].append(boxes[i])
        elif labels[i] == 7:
            res["elements"].append(boxes[i])
    
    # sort boxes
    res["x_ticks"].sort(key=lambda x: x[0])
    res["y_ticks"].sort(key=lambda x: x[1])
    return res


import math


def parse_json4series(json_file: str) -> dict:
    # read annotation
    data = read_json(json_file)
    series = data["data-series"]
    chart_type = data["chart-type"]
    
    x_series = [i["x"] for i in series]
    y_series = [i["y"] for i in series]
    return {"x": x_series, "y": y_series, "chart_type": chart_type}


def get_scale(values: list, boxes: list, axis: int=1):
    m = len(boxes)
    val1, val2 = None, None
    h1, h2 = None, None
    for i in range(m):
        # break early
        if val1 and val2:
            break
            
        if values[i] and not val1:
            val1 = values[i]
            h1 = boxes[i][axis]
        elif values[i] and val1:
            val2 = values[i]
            h2 = boxes[i][axis]
    
    if val1 and val2:
        dv = val1 - val2
        dh = h1 - h2
        return val2, h2, dv/dh
    elif val2 and not val1:
        val1, h1 = val2 * 10, (h2 / 2.5) 
        dv = val1 - val2
        dh = h1 - h2
        return val2, h2, dv/dh
    elif val1 and not val2:
        val2, h2 = val1 * 0.1, (h1 * 2.5) % 1 
        dv = val1 - val2
        dh = h1 - h2
        return val2, h2, dv/dh
    else:
        val1, val2, h1, h2 = 120, 10, 0.25, 0.75
        dv = val1 - val2
        dh = h2 - h1
        return val2, h2, dv/dh

    
def get_value(x: float, scales: tuple):
    v1, h1, dvdh = scales
    return v1 + ((x - h1) * dvdh)
    

def relu(x):
    if x < 0:
        return 0
    else:
        return x
    
    
def dot_series(elements: list):
    # get all in the same column
    # sort by x first
    elements_copy = [i for i in elements]
    groups = []
    while elements_copy != []:
        x1 = elements_copy[0]
        elements_copy.remove(x1)
        temp = [x1,]
        for x2 in elements_copy:
            # if xcenters match, they are in same group
            close = math.isclose(x1[0], x2[0], rel_tol=0.01, abs_tol=0.01)
            if close:
                temp.append(x2)
        
        # removed grouped dots
        elements_copy = [i for i in elements_copy if i not in temp]
        groups.append(temp)
    
    return [len(i) for i in groups]


def is_numerical(x):
    """Test whether input can be parsed as a Python float."""
    try:
        float(x)
        return True
    except ValueError:
        return False
    

def get_series(boxes: list, labels: list, img: Image, reader: Reader):
    res = parse_boxes(boxes, labels)
    
    elements = res["elements"]
    chart_type = res["chart_type"]
    chart_box = res["chart_box"]
    x_boxes = res["x_ticks"]
    y_boxes = res["y_ticks"]
    
    n_elements = len(elements)
    
    try:
        if chart_type == 0:  # vertical_bar
            element_vals = [xywh2xyxy(i) for i in sorted(res["elements"], key=lambda x: x[0])]
            
            x = [words_crop_to_text(expand_box(i), img, reader) for i in x_boxes]

            y_axis = [nums_crop_to_text(expand_box(i), img, c_num=6) for i in y_boxes]   
            scales = get_scale(y_axis, y_boxes, axis=1)
            y = [get_value(x[1], scales) for x in element_vals]

        elif chart_type == 1:   # horizontal_bar
            element_vals = [xywh2xyxy(i) for i in sorted(res["elements"], key=lambda x: x[1])]
            
            y = [words_crop_to_text(expand_box(i), img, reader) for i in y_boxes]
            x_axis = [nums_crop_to_text(expand_box(i), img, c_num=6) for i in x_boxes]
            scales = get_scale(x_axis, x_boxes, axis=0)
            x = [get_value(x[2], scales) for x in element_vals]

        elif chart_type == 2:   # scatter
            element_vals = [xywh2xyxy(i) for i in sorted(res["elements"], key=lambda x: x[0])]
            
            y_axis = [nums_crop_to_text(expand_box(i), img, c_num=6) for i in y_boxes]   
            scales = get_scale(y_axis, y_boxes, axis=1)
            y = [get_value(x[1], scales) for x in element_vals]

            x_axis = [nums_crop_to_text(expand_box(i), img, c_num=6) for i in x_boxes]
            scales = get_scale(x_axis, x_boxes, axis=0)
            x = [get_value(x[2], scales) for x in element_vals]

        elif chart_type == 3:   # dot
            # sort elements
            elements = [i for i in sorted(res["elements"], key=lambda x: x[0])]
            
            y = dot_series(elements)
            try:
                x_axis = [nums_crop_to_text(expand_box(i), img, c_num=6) for i in x_boxes]
                scales = get_scale(x_axis, x_boxes, axis=0)
                x = [get_value(x[2], scales) for x in element_vals]
            except:
                x = [words_crop_to_text(expand_box(i), img, reader) for i in x_boxes]

        elif chart_type == 4:   # line
            element_vals = [xywh2xyxy(i) for i in sorted(res["elements"], key=lambda x: x[0])]
            
            y_axis = [nums_crop_to_text(expand_box(i), img, c_num=6) for i in y_boxes]   
            scales = get_scale(y_axis, y_boxes, axis=1)
            y = [get_value(x[1], scales) for x in element_vals]

            x = [words_crop_to_text(expand_box(i), img, reader) for i in x_boxes]
    except:
        x = ["aeiou"] * n_elements
        y = list(range(n_elements)) 
        
    return {"x": x, "y": y, "chart_type": ID2LABELS.get(chart_type)}


