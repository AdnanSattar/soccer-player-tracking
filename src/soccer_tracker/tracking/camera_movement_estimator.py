"""Camera movement estimation using optical flow."""

import os
import pickle

import cv2
import numpy as np

from ..utils import get_center_of_bbox, get_foot_position


class CameraMovementEstimator:
    """Estimate camera movement between frames using optical flow."""

    def __init__(self, first_frame):
        """Initialize the camera movement estimator.
        
        Args:
            first_frame: First frame of the video (reference frame)
        """
        self.first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Detect features in the first frame
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        self.p0 = cv2.goodFeaturesToTrack(
            self.first_frame,
            mask=None,
            **self.feature_params
        )

    def get_camera_movement(self, video_frames, read_from_stub=False, stub_path=None):
        """Calculate camera movement for each frame.
        
        Args:
            video_frames: List of video frames
            read_from_stub: Whether to read from cached stub file
            stub_path: Path to the stub file
            
        Returns:
            List of camera movement vectors per frame
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                camera_movement_per_frame = pickle.load(f)
            return camera_movement_per_frame

        camera_movement_per_frame = []
        prev_gray = self.first_frame
        prev_points = self.p0

        for frame in video_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            if prev_points is not None and len(prev_points) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, prev_points, None, **self.lk_params
                )
                
                # Select good points
                good_new = p1[st == 1]
                good_old = prev_points[st == 1]
                
                if len(good_new) > 0 and len(good_old) > 0:
                    # Calculate average movement
                    movement = np.mean(good_new - good_old, axis=0)
                    camera_movement_per_frame.append(movement.tolist())
                else:
                    camera_movement_per_frame.append([0.0, 0.0])
                
                # Update points for next iteration
                prev_points = cv2.goodFeaturesToTrack(
                    gray,
                    mask=None,
                    **self.feature_params
                )
            else:
                camera_movement_per_frame.append([0.0, 0.0])
                prev_points = cv2.goodFeaturesToTrack(
                    gray,
                    mask=None,
                    **self.feature_params
                )
            
            prev_gray = gray

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement_per_frame, f)

        return camera_movement_per_frame

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """Adjust track positions based on camera movement.
        
        Args:
            tracks: Dictionary containing tracks
            camera_movement_per_frame: List of camera movement vectors per frame
        """
        cumulative_movement = np.array([0.0, 0.0])
        
        for frame_num, movement in enumerate(camera_movement_per_frame):
            cumulative_movement += np.array(movement)
            
            for object_type, object_tracks in tracks.items():
                if frame_num < len(object_tracks):
                    for track_id, track_info in object_tracks[frame_num].items():
                        if 'position' in track_info:
                            position = np.array(track_info['position'])
                            adjusted_position = position - cumulative_movement
                            tracks[object_type][frame_num][track_id]['position_adjusted'] = adjusted_position.tolist()

    def draw_camera_movement(self, output_video_frames, camera_movement_per_frame):
        """Draw camera movement visualization on frames.
        
        Args:
            output_video_frames: List of video frames
            camera_movement_per_frame: List of camera movement vectors per frame
            
        Returns:
            List of frames with camera movement visualization
        """
        cumulative_movement = np.array([0.0, 0.0])
        
        for frame_num, frame in enumerate(output_video_frames):
            if frame_num < len(camera_movement_per_frame):
                cumulative_movement += np.array(camera_movement_per_frame[frame_num])
                
                # Draw movement vector
                start_point = (50, 50)
                end_point = (
                    int(start_point[0] + cumulative_movement[0] * 10),
                    int(start_point[1] + cumulative_movement[1] * 10)
                )
                
                cv2.arrowedLine(
                    frame,
                    start_point,
                    end_point,
                    (255, 0, 0),
                    3,
                    tipLength=0.3
                )
                
                cv2.putText(
                    frame,
                    f"Camera Movement: ({cumulative_movement[0]:.1f}, {cumulative_movement[1]:.1f})",
                    (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )
        
        return output_video_frames
