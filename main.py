from vehicle_detection import detect_vehicles


video_path = "input/traffic.mp4"


vehicle_count = detect_vehicles(video_path)


print("Final Vehicle Count:", vehicle_count)