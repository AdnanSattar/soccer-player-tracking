"""Object tracking using YOLOv8 and ByteTrack."""

import os
import pickle

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

from ..utils import get_bbox_width, get_center_of_bbox, get_foot_position


class Tracker:
    """Tracker class for detecting and tracking objects in video frames."""

    def __init__(self, model_path, field_boundaries=None, min_bbox_area=500, max_bbox_area=50000):
        """Initialize the tracker with a YOLO model.

        Args:
            model_path: Path to the YOLO model file (.pt)
            field_boundaries: Optional numpy array of field boundary vertices [[x1,y1], [x2,y2], ...]
            min_bbox_area: Minimum bounding box area to filter small detections (default: 500)
            max_bbox_area: Maximum bounding box area to filter large detections (default: 50000)
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.field_boundaries = field_boundaries
        self.min_bbox_area = min_bbox_area
        self.max_bbox_area = max_bbox_area

    def add_position_to_tracks(self, tracks):
        """Add position information to tracks.

        Args:
            tracks: Dictionary containing tracks for players, referees, and ball
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object_type == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_type][frame_num][track_id]["position"] = position

    def interpolate_ball_positions(self, ball_positions):
        """Interpolate missing ball positions.

        Args:
            ball_positions: List of ball position dictionaries

        Returns:
            List of interpolated ball positions
        """
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def is_inside_field(self, bbox):
        """Check if a bounding box is inside the field boundaries.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            True if bbox is inside field, False otherwise
        """
        if self.field_boundaries is None:
            return True  # No filtering if boundaries not set
        
        # Check if the center of the bbox is inside the field
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        center_point = (int(center_x), int(center_y))
        
        # Check if center point is inside the polygon
        result = cv2.pointPolygonTest(self.field_boundaries, center_point, False)
        return result >= 0
    
    def is_valid_bbox_size(self, bbox):
        """Check if bounding box size is within acceptable range.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            True if bbox size is valid, False otherwise
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        return self.min_bbox_area <= area <= self.max_bbox_area

    def detect_frames(self, frames):
        """Detect objects in video frames.

        Args:
            frames: List of video frames

        Returns:
            List of detection results
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            # Increase confidence threshold for better accuracy
            detections_batch = self.model.predict(frames[i : i + batch_size], conf=0.25)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """Get object tracks from video frames.

        Args:
            frames: List of video frames
            read_from_stub: Whether to read from cached stub file
            stub_path: Path to the stub file

        Returns:
            Dictionary containing tracks for players, referees, and ball
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Map COCO classes to soccer classes if using pre-trained model
            # COCO: "person" -> soccer: "player", "sports ball" -> "ball"
            coco_to_soccer = {
                "person": "player",
                "sports ball": "ball",
            }

            # Check if we're using a COCO model (has "person" class)
            is_coco_model = "person" in cls_names_inv

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Map classes: COCO -> Soccer, or handle soccer-specific classes
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                class_name = cls_names[class_id]

                # Convert GoalKeeper to player object (soccer model)
                if class_name == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv.get(
                        "player", class_id
                    )
                # Convert COCO "person" to "player" (pre-trained model)
                elif is_coco_model and class_name == "person":
                    # Keep as person, we'll map it later
                    pass

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                class_name = cls_names[cls_id]

                # Filter: Check if bbox is inside field and has valid size
                if not self.is_inside_field(bbox):
                    continue  # Skip detections outside field (audience, etc.)
                
                if not self.is_valid_bbox_size(bbox):
                    continue  # Skip detections with invalid size

                # Handle both soccer-specific and COCO models
                if class_name == "player" or (is_coco_model and class_name == "person"):
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif class_name == "referee":
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                class_name = cls_names[cls_id]

                # Filter: Check if bbox is inside field and has valid size
                if not self.is_inside_field(bbox):
                    continue  # Skip detections outside field
                
                if not self.is_valid_bbox_size(bbox):
                    continue  # Skip detections with invalid size

                # Handle both soccer-specific and COCO models
                if class_name == "ball" or (is_coco_model and class_name == "sports ball"):
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """Draw an ellipse around an object.

        Args:
            frame: Video frame
            bbox: Bounding box coordinates
            color: Color of the ellipse
            track_id: Optional track ID to display

        Returns:
            Frame with ellipse drawn
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        """Draw a triangle above an object.

        Args:
            frame: Video frame
            bbox: Bounding box coordinates
            color: Color of the triangle

        Returns:
            Frame with triangle drawn
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array(
            [
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20],
            ]
        )
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """Draw team ball control statistics.

        Args:
            frame: Video frame
            frame_num: Current frame number
            team_ball_control: Array of team ball control per frame

        Returns:
            Frame with ball control statistics drawn
        """
        # Draw a semi-transparent rectangle (optimized to avoid full frame copy)
        # Draw rectangle directly with alpha blending on ROI only
        roi = frame[850:970, 1350:1900]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (550, 120), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

        team_ball_control_till_frame = team_ball_control[: frame_num + 1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames > 0:
            team_1 = team_1_num_frames / total_frames
            team_2 = team_2_num_frames / total_frames
        else:
            team_1 = 0.0
            team_2 = 0.0

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1 * 100:.2f}%",
            (1400, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2 * 100:.2f}%",
            (1400, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """Draw annotations on video frames.

        Args:
            video_frames: List of video frames
            tracks: Dictionary containing tracks
            team_ball_control: Array of team ball control per frame

        Returns:
            List of annotated video frames
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            # Use copy only if we need to modify, otherwise work in-place where possible
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
