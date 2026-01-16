"""Video processing utility functions."""

import cv2


def read_video(video_path):
    """Read video file and return all frames.

    Args:
        video_path: Path to the video file

    Returns:
        List of video frames (numpy arrays)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path):
    """Save video frames to a video file.

    Args:
        output_video_frames: List of video frames (numpy arrays)
        output_video_path: Path to save the output video
    """
    if not output_video_frames:
        raise ValueError("No frames to save")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    height, width = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, 24.0, (width, height))

    for frame in output_video_frames:
        out.write(frame)

    out.release()


def save_video_streaming(
    video_frames,
    tracks,
    team_ball_control,
    camera_movement_per_frame,
    tracker,
    camera_movement_estimator,
    speed_and_distance_estimator,
    output_video_path,
):
    """Save video by processing frames one at a time (memory efficient).

    Args:
        video_frames: List of input video frames
        tracks: Dictionary containing tracks
        team_ball_control: Array of team ball control per frame
        camera_movement_per_frame: List of camera movement vectors
        tracker: Tracker instance
        camera_movement_estimator: CameraMovementEstimator instance
        speed_and_distance_estimator: SpeedAndDistanceEstimator instance
        output_video_path: Path to save the output video
    """
    if not video_frames:
        raise ValueError("No frames to save")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    height, width = video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, 24.0, (width, height))

    total_frames = len(video_frames)
    num_track_frames = len(tracks["players"])

    if num_track_frames != total_frames:
        print(
            f"⚠ Warning: Video has {total_frames} frames but tracks have {num_track_frames} frames"
        )
        print(f"  Processing {min(total_frames, num_track_frames)} frames...")

    for frame_num in range(min(total_frames, num_track_frames)):
        if (frame_num + 1) % 100 == 0:
            print(f"  Processing frame {frame_num+1}/{min(total_frames, num_track_frames)}...")

        frame = video_frames[frame_num].copy()

        # Get tracks for this frame (with bounds checking)
        if frame_num >= len(tracks["players"]):
            # No tracks for this frame, skip annotations
            out.write(frame)
            continue

        # Safely get track dictionaries for this frame
        try:
            player_dict = tracks["players"][frame_num] if frame_num < len(tracks["players"]) else {}
        except (IndexError, KeyError):
            player_dict = {}

        try:
            ball_dict = tracks["ball"][frame_num] if frame_num < len(tracks["ball"]) else {}
        except (IndexError, KeyError):
            ball_dict = {}

        try:
            referee_dict = (
                tracks["referees"][frame_num] if frame_num < len(tracks["referees"]) else {}
            )
        except (IndexError, KeyError):
            referee_dict = {}

        # Draw Players
        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            frame = tracker.draw_ellipse(frame, player["bbox"], color, track_id)
            if player.get("has_ball", False):
                frame = tracker.draw_triangle(frame, player["bbox"], (0, 0, 255))

        # Draw Referee
        for _, referee in referee_dict.items():
            frame = tracker.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

        # Draw ball (handle case where ball_dict might be empty or have different structure)
        if ball_dict:
            for track_id, ball in ball_dict.items():
                if isinstance(ball, dict) and "bbox" in ball:
                    frame = tracker.draw_triangle(frame, ball["bbox"], (0, 255, 0))

        # Draw Team Ball Control (only if we have data for this frame)
        if frame_num < len(team_ball_control):
            frame = tracker.draw_team_ball_control(frame, frame_num, team_ball_control)

        # Draw Camera movement
        if frame_num < len(camera_movement_per_frame):
            import numpy as np

            cumulative_movement = np.array([0.0, 0.0])
            for i in range(frame_num + 1):
                if i < len(camera_movement_per_frame):
                    cumulative_movement += np.array(camera_movement_per_frame[i])
            # Simple camera movement visualization
            start_point = (50, 50)
            end_point = (
                int(start_point[0] + cumulative_movement[0] * 10),
                int(start_point[1] + cumulative_movement[1] * 10),
            )
            cv2.arrowedLine(frame, start_point, end_point, (255, 0, 0), 3, tipLength=0.3)

        # Draw Speed and Distance
        for object_type, object_tracks in tracks.items():
            if object_type == "ball" or object_type == "referees":
                continue
            if frame_num < len(object_tracks):
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get("speed", None)
                        distance = track_info.get("distance", None)
                        if speed is not None and distance is not None:
                            from ..utils import get_foot_position

                            bbox = track_info["bbox"]
                            position = get_foot_position(bbox)
                            position = list(position)
                            position[1] += 40
                            position = tuple(map(int, position))
                            cv2.putText(
                                frame,
                                f"{speed:.2f} km/h",
                                position,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 0),
                                2,
                            )
                            cv2.putText(
                                frame,
                                f"{distance:.2f} m",
                                (position[0], position[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 0),
                                2,
                            )

        out.write(frame)

    out.release()
