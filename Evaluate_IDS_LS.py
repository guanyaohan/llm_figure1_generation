
import math, cv2, numpy as np, torch
from PIL import Image


import easyocr


def load_and_crop(path, white_thr: int = 250):

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < white_thr))
    if coords.size == 0:                      
        return img, img.shape[0] * img.shape[1]
    x, y, w, h = cv2.boundingRect(coords)
    img_cropped = img[y:y + h, x:x + w]
    area = img_cropped.shape[0] * img_cropped.shape[1]
    return img_cropped, area


def iou(box_a, box_b):
    """IoU of two boxes [x1,y1,x2,y2]."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter_area / (area_a + area_b - inter_area + 1e-9)


def nms(boxes, iou_th: float = 0.5):

    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]             
    keep_idx = []
    while order.size:
        i = order[0]
        keep_idx.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou_vals = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou_vals < iou_th]
    return boxes[keep_idx].tolist()



def count_objects_sam_yolo(img_bgr, sam_ckpt="sam_vit_h_4b8939.pth",
                           yolo_ckpt="yolov9c.pt", device="cuda"):
    """返回 distinct visual objects 数量。"""
    # ----- SAM -----
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(img_bgr[..., ::-1])    # BGR→RGB
    masks, _, _ = predictor.predict(
        point_coords=None, point_labels=None, multimask_output=True)
    sam_boxes = []
    for m in masks:
        ys, xs = np.where(m)
        if ys.size == 0:
            continue
        sam_boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    # ----- YOLO -----
    from ultralytics import YOLO
    yolo = YOLO(yolo_ckpt)
    yolo_results = yolo(img_bgr, verbose=False)[0]
    yolo_boxes = yolo_results.boxes.xyxy.cpu().numpy().tolist()

    all_boxes = sam_boxes + yolo_boxes
    uniq_boxes = nms(all_boxes, iou_th=0.5)
    return len(uniq_boxes)


def count_textboxes_donut(img_bgr, device="cuda"):

    from transformers import DonutProcessor, VisionEncoderDecoderModel
    processor = DonutProcessor.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-docvqa")
    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-docvqa").to(device)
    img_pil = Image.fromarray(img_bgr[..., ::-1])   # BGR→RGB PIL
    inputs = processor(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        encoder_out = model.encoder(inputs.pixel_values).last_hidden_state

    tokens = processor.tokenizer.decode(
        encoder_out.argmax(dim=-1)[0], skip_special_tokens=True)
    return tokens.count("\"bbox\"")



def saliency_bonus(img_bgr, device="cuda"):
  
    try:
        from deepgaze_pytorch import get_deepgaze
    except ImportError:
        return 0.0
    model_sal = get_deepgaze('deepgaze_iii').to(device).eval()
    with torch.no_grad():
        smap = model_sal(Image.fromarray(img_bgr[..., ::-1]))[0, 0]
    p = smap.cpu().numpy()
    p = p / p.sum()
    H = -np.nansum(p * np.log2(p + 1e-9))
    H_norm = H / math.log2(p.size)           # 0–1
    return 0.05 * (1 - H_norm)              




def info_density_score(image_path: str,
                       use_saliency: bool = False,
                       sam_ckpt="sam_vit_h_4b8939.pth",
                       yolo_ckpt="yolov9c.pt",
                       device="cuda"):
    """Compute IDS for a single figure."""
    img, area = load_and_crop(image_path)
    objects = count_objects_sam_yolo(img, sam_ckpt, yolo_ckpt, device)
    text_boxes = count_textboxes_donut(img, device)
    ids_core = (objects + 0.5 * text_boxes) / math.sqrt(area)
    bonus = saliency_bonus(img, device) if use_saliency else 0.0
    return round(ids_core + bonus, 4)


reader = easyocr.Reader(['en'], gpu=False)

# ------------------------- 6. Legibility Score -------------------------
def get_text_bboxes(img_bgr):
    result = reader.readtext(img_bgr[..., ::-1], detail=1, paragraph=False)
    # result: list[(bbox, text, conf)]
    bboxes = []
    for (pts, _, conf) in result:
        if conf < 0.4:
            continue
        pts = np.array(pts, dtype=int)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        bboxes.append([x1, y1, x2, y2])
    return bboxes

def relative_luminance(rgb):

    rgb = rgb / 255.0
    mask = rgb <= 0.04045
    rgb[mask] /= 12.92
    rgb[~mask] = ((rgb[~mask] + 0.055) / 1.055) ** 2.4
    R, G, B = rgb
    return 0.2126 * R + 0.7152 * G + 0.0722 * B


def bbox_contrast(img_bgr, bbox):

    x1, y1, x2, y2 = bbox
    patch = img_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return 1.0

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).flatten()
    thresh = np.percentile(gray, 20)
    text_mask = gray <= thresh
    bg_mask   = gray >  thresh
    if text_mask.sum() == 0 or bg_mask.sum() == 0:
        return 1.0
    text_rgb = patch.reshape(-1,3)[text_mask].mean(axis=0)
    bg_rgb   = patch.reshape(-1,3)[bg_mask].mean(axis=0)
    Lt = relative_luminance(text_rgb)
    Lb = relative_luminance(bg_rgb)
    Lmax, Lmin = max(Lt, Lb), min(Lt, Lb)
    return (Lmax + 0.05) / (Lmin + 0.05)


def legibility_score(image_path, dpi=300):
    """Compute LS for a single figure."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)


    bboxes = get_text_bboxes(img)
    if not bboxes:
        return 1.0    


    heights_px = [y2 - y1 for _,y1,_,y2 in bboxes]
    h_min = np.percentile(heights_px, 20)         
    pt = 72 * h_min / dpi
    f_font = max(0.0, min(1.0, (pt - 6) / 6))     


    contrasts = [bbox_contrast(img, bb) for bb in bboxes]
    C = np.percentile(contrasts, 20)              
    if C < 3:
        f_contrast = 0.0
    elif C < 4.5:
        f_contrast = (C - 3) / 1.5
    else:
        f_contrast = 1.0

   
    LS = 0.5 * f_font + 0.5 * f_contrast
    return round(LS, 4)


if __name__ == "__main__":
    import sys, time
    if len(sys.argv) < 2:
        print("Usage: python ids.py <figure.png>")
        sys.exit(0)

    img_path = sys.argv[1]

    t0 = time.time()
    ids = info_density_score(img_path, use_saliency=False)
    ls  = legibility_score(img_path)
    print(f"IDS = {ids:.4f}   LS = {ls:.4f}   (elapsed {time.time()-t0:.2f}s)")


