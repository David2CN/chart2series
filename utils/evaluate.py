import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rapidfuzz.distance.Levenshtein import distance as levenshtein
from sklearn.metrics import r2_score

    
def process_single_result(result, img: Image, reader: Reader)-> dict:
    # get boxes and labels
    boxes = result.boxes.xywhn.tolist()
    labels = result.boxes.cls.tolist()

    # get dataseries
    dataseries = get_series(boxes, labels, img, reader)
    return dataseries


def predict_one_batch(batch_images: list, model: YOLO, reader: Reader) -> tuple:
    results = model(batch_images, agnostic_nms=True, iou=0.45, verbose=False)
    ids = []
    data_series = []
    chart_types = []
    i = 0
    for result in results:
        img_name = Path(batch_images[i]).stem
        ids.extend([f"{img_name}_x", f"{img_name}_y"])
        
        img = read_image(batch_images[i])
        res = process_single_result(result, img=img, reader=reader)
        
        x = ";".join([str(i) for i in res["x"]])
        y = ";".join([str(i) for i in res["y"]])
        
        chart_type = res["chart_type"]
        data_series.extend([x, y])
        chart_types.extend([chart_type, chart_type])
        i += 1
    
    return ids, data_series, chart_types


def get_batches(images: list, batch_size: int=8):
    if batch_size == 1:
        return [[i] for i in images]
    else:
        batches = []
        num_batches = len(images) // batch_size
        for i in range(num_batches):
            batches.append(images[i*batch_size: (i+1)*batch_size])
        
        # add any leftovers
        temp = images[num_batches*batch_size:]
        if temp != []:
            batches.append(temp)
        return batches


def create_pred_df(test_dir: str, model: YOLO, reader: Reader, 
                   batch_size: int=8, limit: int=None) -> pd.DataFrame:
    test_images = sorted([str(i) for i in Path(test_dir).glob("*.jpg")])[:limit]
    batched_test_images = get_batches(test_images, batch_size=batch_size)
    
    ids = []
    data_series = []
    chart_types = []
    for batch in tqdm(batched_test_images):
        batch_ids, batch_series, batch_types = predict_one_batch(batch, model, reader)
        ids.extend(batch_ids)
        data_series.extend(batch_series)
        chart_types.extend(batch_types)
    
    df = pd.DataFrame({
        "id": ids,
        "data_series": data_series,
        "chart_type": chart_types
    })
    return df


def create_pred_df2(test_dir: str, model: YOLO, reader: Reader, 
                   batch_size: int=8, limit: int=None) -> pd.DataFrame:
    test_images = sorted([str(i) for i in Path(test_dir).glob("*.jpg")])[:limit]
    batched_test_images = get_batches(test_images, batch_size=batch_size)
    
    ids = []
    data_series = []
    chart_types = []
    for batch in tqdm(batched_test_images):
        batch_ids, batch_series, batch_types = predict_one_batch(batch, model, reader)
        ids.extend(batch_ids)
        data_series.extend(batch_series)
        chart_types.extend(batch_types)
        
    df = pd.DataFrame({
        "data_series": data_series,
        "chart_type": chart_types
    }, index=ids)
    return df


def create_groundtruth_df(test_dir: str, limit: int=None) -> pd.DataFrame:
    file_names = sorted([str(i) for i in Path(test_dir).glob("*.json")])[:limit]
    ids = []
    data_series = []
    chart_types = []
    for json_file in tqdm(file_names):
        name = Path(json_file).stem
        ids.extend([f"{name}_x", f"{name}_y"])
        
        res = parse_json4series(json_file)
        
        x = ";".join([str(i) for i in res["x"]])
        y = ";".join([str(i) for i in res["y"]])
        chart_type = res["chart_type"]
        data_series.extend([x, y])
        chart_types.extend([chart_type, chart_type])
    
    df = pd.DataFrame({
        "data_series": data_series,
        "chart_type": chart_types
    }, index=ids)
    return df


def sigmoid(x):
    return 2 - 2 / (1 + np.exp(-x))


def normalized_rmse(y_true, y_pred):
    # The argument to the sigmoid transform is equal to 
    # rmse(y_true, y_pred) / rmse(y_true, np.mean(y_true))
    return sigmoid((1 - r2_score(y_true, y_pred)) ** 0.5)


def normalized_levenshtein_score(y_true, y_pred):
    total_distance = np.sum([levenshtein(yt, yp) for yt, yp in zip(y_true, y_pred)])
    length_sum = np.sum([len(yt) for yt in y_true])
    return sigmoid(total_distance / length_sum)


def score_series(y_true, y_pred):
    if len(y_true) != len(y_pred):
        return 0.0
    if isinstance(y_true[0], str):
        return normalized_levenshtein_score(y_true, y_pred)
    else:
        return normalized_rmse(y_true, y_pred)


def benetech_score(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """Evaluate predictions using the metric from the Benetech - Making Graphs Accessible.
    
    Parameters
    ----------
    ground_truth: pd.DataFrame
        Has columns `[data_series, chart_type]` and an index `id`. Values in `data_series` 
        should be either arrays of floats or arrays of strings.
    
    predictions: pd.DataFrame
    """
    if not ground_truth.index.equals(predictions.index):
        raise ValueError("Must have exactly one prediction for each ground-truth instance.")
    if not ground_truth.columns.equals(predictions.columns):
        raise ValueError(f"Predictions must have columns: {ground_truth.columns}.")
    pairs = zip(ground_truth.itertuples(index=False), predictions.itertuples(index=False))
    scores = []
    for (gt_series, gt_type), (pred_series, pred_type) in pairs:
        if gt_type != pred_type:  # Check chart_type condition
            scores.append(0.0)
        else:  # Score with RMSE or Levenshtein as appropriate
            scores.append(score_series(gt_series, pred_series))
    return np.mean(scores)