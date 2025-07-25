def get_center_of_bbox(bbox):
    # bbox can be [x1, y1, x2, y2] or {'x1':..., 'y1':..., 'x2':..., 'y2':...}
    if isinstance(bbox, dict):
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    else:
        x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)

def measure_distance(p1, p2):
    # Support both (x, y) tuples/lists and {'x':..., 'y':...} dicts
    if isinstance(p1, dict):
        x1, y1 = p1['x'], p1['y']
    else:
        x1, y1 = p1
    if isinstance(p2, dict):
        x2, y2 = p2['x'], p2['y']
    else:
        x2, y2 = p2
    
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5