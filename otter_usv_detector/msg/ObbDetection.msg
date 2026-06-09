# Single Oriented Bounding Box detection result.
# Coordinates are in image pixels (not normalised).

string  class_name        # detected class label
float32 confidence        # detection confidence [0, 1]

# OBB centre and geometry (pixel coordinates)
float32 cx                # centre x
float32 cy                # centre y
float32 width             # OBB width  (longer side before rotation)
float32 height            # OBB height (shorter side before rotation)
float32 angle_deg         # rotation angle in degrees (Ultralytics convention)

# Four corners in pixel coordinates: [x1,y1, x2,y2, x3,y3, x4,y4]
# Corner order matches Ultralytics obb.xyxyxyxy (clockwise from top-left)
float32[8] corners
