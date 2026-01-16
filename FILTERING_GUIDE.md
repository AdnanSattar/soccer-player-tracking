# 🎯 Field Boundary Filtering Guide

Improve tracking accuracy by filtering out audience members and false detections.

## Problem

When using a pre-trained YOLOv8 model (like `yolov8n.pt`), the tracker detects all people in the video, including:

- Audience members in the stands/pavilion
- People outside the field boundaries
- False detections

This results in poor tracking accuracy.

## Solution

The tracker now includes **field boundary filtering** to exclude detections outside the soccer field.

## How It Works

1. **Field Boundary Detection**: Uses the `ViewTransformer` pixel vertices to define the field boundaries
2. **Bounding Box Filtering**: Only tracks objects whose center point is inside the field polygon
3. **Size Filtering**: Filters out detections that are too small (audience far away) or too large (false detections)

## Configuration

### Adjust Field Boundaries

If the field boundaries don't match your video, update them in `main.py`:

```python
# The ViewTransformer defines the field boundaries
view_transformer = ViewTransformer()
# Update pixel_vertices in view_transformer.py to match your video
```

Or directly in `src/soccer_tracker/analysis/view_transformer.py`:

```python
self.pixel_vertices = np.array([
    [489, 200],   # Top-left corner of field
    [236, 813],   # Bottom-left corner
    [2193, 813],  # Bottom-right corner
    [1960, 180]   # Top-right corner
])
```

### Adjust Bounding Box Size Filters

In `main.py`, adjust the min/max bbox area:

```python
tracker = Tracker(
    model_path,
    field_boundaries=field_boundaries,
    min_bbox_area=800,   # Increase to exclude smaller detections (audience)
    max_bbox_area=40000  # Decrease to exclude larger false detections
)
```

**Guidelines:**

- **min_bbox_area**:
  - Too low: Includes audience members
  - Too high: Excludes actual players
  - Start with 500-1000 for 1080p video
  
- **max_bbox_area**:
  - Too high: Includes false detections
  - Too low: Excludes players close to camera
  - Start with 30000-50000 for 1080p video

### Adjust Confidence Threshold

In `src/soccer_tracker/tracking/tracker.py`, the `detect_frames` method:

```python
# Current: conf=0.25 (25% confidence)
detections_batch = self.model.predict(frames[i : i + batch_size], conf=0.25)

# Increase to 0.3-0.4 for fewer false positives
# Decrease to 0.15-0.2 for more detections (but more false positives)
```

## Clearing Stub Files

If you update the filtering parameters, you need to clear the stub files to recalculate:

```bash
# Delete tracking stub
rm model/track_stubs.pkl

# Delete camera movement stub (optional, only if you changed field boundaries)
rm model/camera_movement_stub.pkl
```

Or on Windows:

```powershell
del model\track_stubs.pkl
del model\camera_movement_stub.pkl
```

## Testing Filtering

1. Run the tracker and check the output video
2. If audience is still being tracked:
   - Increase `min_bbox_area` (e.g., 1000-1500)
   - Verify field boundaries are correct
   - Increase confidence threshold
3. If players are being excluded:
   - Decrease `min_bbox_area` (e.g., 500-800)
   - Verify field boundaries include all players
   - Decrease confidence threshold

## Best Practices

1. **Use a custom trained model**: Train YOLOv8 on soccer-specific data for best results
2. **Calibrate field boundaries**: Use the first frame to manually set field corners
3. **Fine-tune size filters**: Adjust based on your camera angle and video resolution
4. **Monitor output**: Check the output video to verify filtering is working correctly

## Troubleshooting

### Still tracking audience?

- Field boundaries might be too large - check `pixel_vertices`
- `min_bbox_area` might be too low - increase it
- Confidence threshold might be too low - increase to 0.3-0.4

### Missing players?

- Field boundaries might be too small - expand `pixel_vertices`
- `min_bbox_area` might be too high - decrease it
- Confidence threshold might be too high - decrease to 0.2-0.25

### No detections?

- Check if field boundaries are correct
- Verify video file is readable
- Check model file exists and is valid
