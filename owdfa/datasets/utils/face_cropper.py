try:
    import dlib
except:
    print('Please install dlib when using face detection!')


def add_face_margin(x, y, w, h, margin):
    assert margin >= 1.0, "margin must be greater than 1.0"
    margin -= 1
    x0 = x - int(w * margin / 2)
    y0 = y - int(h * margin / 2)
    x1 = x + w + int(w * margin / 2)
    y1 = y + h + int(h * margin / 2)
    return [x0, y0, x1, y1]


def get_default_bbox(img, face_boxes, margin):
    x0, y0, x1, y1 = face_boxes
    x0, y0, x1, y1 = add_face_margin(x0, y0, x1-x0, y1-y0, margin)
    max_h, max_w = img.shape[:2]
    x0 = int(max(0, x0))
    y0 = int(max(0, y0))
    x1 = int(min(max_w, x1))
    y1 = int(min(max_h, y1))
    return [x0, y0, x1, y1]


def dlib_crop_face(img, detector, predictor, align=False, margin=1.2):
    dets = detector(img, 0)
    if len(dets) > 0:
        det = dets[0]

        if align:
            shape = predictor(img, det)
            face = dlib.get_face_chip(img, shape, size=320)
        else:
            bbox = [det.left(), det.top(), det.right(), det.bottom()]
            x0, y0, x1, y1 = get_default_bbox(img, bbox, margin=margin)
            assert x0 < x1 and y0 < y1
            face = img[y0:y1, x0:x1]

        img = face
    return img
