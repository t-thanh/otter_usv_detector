# Array of OBB detections for one image frame.

std_msgs/Header   header        # timestamp + frame_id from the source image
uint32            image_width   # source image width  (px)
uint32            image_height  # source image height (px)
ObbDetection[]    detections    # all detections in this frame (may be empty)
