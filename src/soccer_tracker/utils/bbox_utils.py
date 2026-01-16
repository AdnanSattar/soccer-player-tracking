"""Bounding box utility functions."""


def get_center_of_bbox(bbox):
    """Get the center point of a bounding box.
    
    Args:
        bbox: List or tuple of [x1, y1, x2, y2]
        
    Returns:
        Tuple of (x_center, y_center)
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    """Get the width of a bounding box.
    
    Args:
        bbox: List or tuple of [x1, y1, x2, y2]
        
    Returns:
        Width of the bounding box
    """
    return bbox[2] - bbox[0]


def measure_distance(p1, p2):
    """Calculate Euclidean distance between two points.
    
    Args:
        p1: Tuple of (x1, y1)
        p2: Tuple of (x2, y2)
        
    Returns:
        Euclidean distance
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def measure_xy_distance(p1, p2):
    """Calculate x and y distance between two points.
    
    Args:
        p1: Tuple of (x1, y1)
        p2: Tuple of (x2, y2)
        
    Returns:
        Tuple of (x_distance, y_distance)
    """
    return p1[0] - p2[0], p1[1] - p2[1]


def get_foot_position(bbox):
    """Get the foot position (bottom center) of a bounding box.
    
    Args:
        bbox: List or tuple of [x1, y1, x2, y2]
        
    Returns:
        Tuple of (x_center, y_bottom)
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)
