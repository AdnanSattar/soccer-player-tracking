"""Transform camera view to top-down court view."""

import cv2
import numpy as np


class ViewTransformer:
    """Transform points from camera view to top-down court coordinates."""

    def __init__(self, court_width=34, court_length=23.32):
        """Initialize the view transformer.
        
        Args:
            court_width: Width of the court in meters
            court_length: Length of the court in meters
        """
        self.court_width = court_width
        self.court_length = court_length

        # Pixel coordinates of court corners in the video
        self.pixel_vertices = np.array([
            [489, 200],
            [236, 813],
            [2193, 813],
            [1960, 180]
        ])

        # Target coordinates in meters (top-down view)
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point):
        """Transform a point from pixel coordinates to court coordinates.
        
        Args:
            point: Point coordinates [x, y]
            
        Returns:
            Transformed point coordinates or None if outside court
        """
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(
            reshaped_point, self.perspective_transformer
        )
        return transform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        """Add transformed positions to tracks.
        
        Args:
            tracks: Dictionary containing tracks
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position_adjusted = track_info.get('position_adjusted')
                    if position_adjusted is None:
                        continue
                    
                    position = np.array(position_adjusted)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object_type][frame_num][track_id]['position_transformed'] = position_transformed
