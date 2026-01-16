"""Main entry point for Soccer Player Tracking System."""

import os
import sys

import cv2
import numpy as np

from soccer_tracker.analysis import (
    PlayerBallAssigner,
    SpeedAndDistanceEstimator,
    TeamAssigner,
    ViewTransformer,
)
from soccer_tracker.tracking import CameraMovementEstimator, Tracker
from soccer_tracker.utils import read_video, save_video


def main():
    """Main function to process soccer video and track players."""
    # Configuration
    video_path = "input_videos/footbal_match.mp4"
    model_path = "model/best.pt"
    track_stub_path = "model/track_stubs.pkl"
    camera_stub_path = "model/camera_movement_stub.pkl"
    output_path = "output_videos/output_video.mp4"

    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        print("Please ensure the video file exists in the input_videos/ directory.")
        sys.exit(1)

    # Check if model file exists, if not use pre-trained model
    if not os.path.exists(model_path):
        print(f"⚠ Model file not found: {model_path}")
        print("\nUsing pre-trained YOLOv8n model (general-purpose, not soccer-specific).")
        print("For best results, train your own model using Finetuning_yolo_model.ipynb")
        print("or download a soccer-specific model.\n")

        # Use pre-trained model (YOLO will auto-download if needed)
        model_path = "yolov8n.pt"  # YOLO will handle the download

        # Check if model directory exists
        model_dir = os.path.dirname("model/best.pt")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            print(f"Created model directory: {model_dir}")
    else:
        print(f"✓ Using custom model: {model_path}")

    print("=" * 60)
    print("Soccer Player Tracking System")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    if model_path.startswith("yolov8"):
        print("⚠ Using pre-trained model - may not detect soccer-specific objects well")
        print("  Consider training a custom model for better results")
    print("=" * 60)

    # Read the video
    print("\n[1/9] Reading video...")
    try:
        video_frames = read_video(video_path)
        print(f"✓ Loaded {len(video_frames)} frames")
    except Exception as e:
        print(f"✗ Error reading video: {e}")
        sys.exit(1)

    # View Transformer (needed for field boundaries)
    view_transformer = ViewTransformer()
    field_boundaries = view_transformer.pixel_vertices.astype(np.int32)

    # Init Tracker with field boundary filtering
    print("\n[2/9] Initializing tracker...")
    try:
        # Filter settings: min/max bbox area to exclude audience and very small detections
        # Adjust these based on your video resolution and camera angle
        tracker = Tracker(
            model_path,
            field_boundaries=field_boundaries,
            min_bbox_area=800,  # Minimum area (adjust based on your video)
            max_bbox_area=40000,  # Maximum area (to exclude very large false detections)
        )
        print("✓ Tracker initialized with field boundary filtering")
        print(f"  Field boundaries: {len(field_boundaries)} vertices")
        print(f"  Bbox area filter: {tracker.min_bbox_area} - {tracker.max_bbox_area} pixels²")
    except Exception as e:
        print(f"✗ Error initializing tracker: {e}")
        sys.exit(1)

    print("\n[3/9] Detecting and tracking objects...")
    print("  Note: Field boundary filtering is active to exclude audience")
    print("  If you updated field boundaries, delete stub files to recalculate")
    try:
        tracks = tracker.get_object_tracks(
            video_frames, read_from_stub=True, stub_path=track_stub_path
        )
        print("✓ Object tracking completed")
    except Exception as e:
        print(f"✗ Error during object tracking: {e}")
        sys.exit(1)

    print("\n[4/9] Adding positions to tracks...")
    try:
        tracker.add_position_to_tracks(tracks)
        print("✓ Positions added")
    except Exception as e:
        print(f"✗ Error adding positions: {e}")
        sys.exit(1)

    # Camera movement estimator
    print("\n[5/9] Estimating camera movement...")
    try:
        camera_movement_estimator = CameraMovementEstimator(video_frames[0])
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
            video_frames, read_from_stub=True, stub_path=camera_stub_path
        )
        camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
        print("✓ Camera movement estimated")
    except Exception as e:
        print(f"✗ Error estimating camera movement: {e}")
        sys.exit(1)

    # View Transformer (already initialized above)
    print("\n[6/9] Transforming view coordinates...")
    try:
        view_transformer.add_transformed_position_to_tracks(tracks)
        print("✓ View transformation completed")
    except Exception as e:
        print(f"✗ Error in view transformation: {e}")
        sys.exit(1)

    # Interpolate ball positions (for when ball is not detected in the frame)
    print("\n[7/9] Interpolating ball positions...")
    try:
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
        print("✓ Ball positions interpolated")
    except Exception as e:
        print(f"✗ Error interpolating ball positions: {e}")
        sys.exit(1)

    # Speed and distance estimator
    print("\n[8/9] Calculating speed and distance...")
    try:
        speed_and_distance_estimator = SpeedAndDistanceEstimator()
        speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
        print("✓ Speed and distance calculated")
    except Exception as e:
        print(f"✗ Error calculating speed/distance: {e}")
        sys.exit(1)

    # Assign Player Teams
    print("\n[9/9] Assigning teams and ball possession...")
    try:
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

        for frame_num, player_track in enumerate(tracks["players"]):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(
                    video_frames[frame_num], track["bbox"], player_id
                )

                tracks["players"][frame_num][player_id]["team"] = team
                tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[
                    team
                ]

        # Assign ball to player
        player_assigner = PlayerBallAssigner()
        team_ball_control = []
        for frame_num, player_track in enumerate(tracks["players"]):
            ball_bbox = tracks["ball"][frame_num][1]["bbox"]
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1:
                tracks["players"][frame_num][assigned_player]["has_ball"] = True
                team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
            elif assigned_player == -1 and frame_num < 2:
                team_ball_control.append(2)
            else:
                team_ball_control.append(team_ball_control[-1])

        team_ball_control = np.array(team_ball_control)
        print("✓ Teams assigned and ball possession tracked")
    except Exception as e:
        print(f"✗ Error assigning teams: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Draw Output - Process frame by frame to save memory
    print("\n[Drawing annotations...]")
    try:
        # Use streaming approach to process frames one at a time
        from soccer_tracker.utils.video_utils import save_video_streaming

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_video_streaming(
            video_frames,
            tracks,
            team_ball_control,
            camera_movement_per_frame,
            tracker,
            camera_movement_estimator,
            speed_and_distance_estimator,
            output_path,
        )
        print("✓ Annotations drawn and video saved")
    except Exception as e:
        print(f"✗ Error drawing annotations: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("=" * 60)
    print("✓ Processing completed successfully!")
    print(f"✓ Output video saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
