import cv2

from traffic_density import calculate_density
from signal_control import get_signal_time, get_signal_status


# YOLOv3 files
CONFIG_FILE = "models/yolov3.cfg"
WEIGHTS_FILE = "models/yolov3.weights"
NAMES_FILE = "models/coco.names"


# Load class names
with open(NAMES_FILE, "r") as f:
    classes = [line.strip() for line in f.readlines()]


# Vehicle classes
vehicle_classes = ["car", "bus", "truck", "motorbike"]


# Load YOLOv3
net = cv2.dnn.readNetFromDarknet(
    CONFIG_FILE,
    WEIGHTS_FILE
)


# Get output layers
layer_names = net.getLayerNames()

output_layers = [
    layer_names[i - 1]
    for i in net.getUnconnectedOutLayers().flatten()
]


def detect_vehicles(video_path):

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Video could not be opened.")
        return 0

    total_vehicle_count = 0
    frame_count = 0

    while True:

        ret, frame = video.read()

        if not ret:
            break

        height, width = frame.shape[:2]

        # YOLO input
        blob = cv2.dnn.blobFromImage(
            frame,
            1 / 255.0,
            (416, 416),
            swapRB=True,
            crop=False
        )

        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        # Process detections
        for output in outputs:

            for detection in output:

                scores = detection[5:]
                class_id = scores.argmax()
                confidence = scores[class_id]

                if confidence > 0.5:

                    class_name = classes[class_id]

                    if class_name in vehicle_classes:

                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)

                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        # Remove duplicate detections
        indexes = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            0.5,
            0.4
        )

        vehicle_count = 0

        if len(indexes) > 0:

            for i in indexes.flatten():

                x, y, w, h = boxes[i]

                label = classes[class_ids[i]]

                vehicle_count += 1

                # Bounding box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # Vehicle label
                cv2.putText(
                    frame,
                    label,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Average vehicle count
        total_vehicle_count += vehicle_count
        frame_count += 1

        average_vehicle_count = int(
            total_vehicle_count / frame_count
        )

        # Traffic density
        density = calculate_density(
            average_vehicle_count
        )

        # Dynamic green signal time
        signal_time = get_signal_time(
            density
        )

        # Traffic status
        status = get_signal_status(
            density
        )

        # Dynamic signal state
        signal_state = "GREEN"

        # Display vehicle count
        cv2.putText(
            frame,
            f"Vehicles: {vehicle_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # Display average count
        cv2.putText(
            frame,
            f"Average: {average_vehicle_count}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # Display density
        cv2.putText(
            frame,
            f"Density: {density}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # Display signal time
        cv2.putText(
            frame,
            f"Green Time: {signal_time} sec",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # Display traffic status
        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # Display signal state
        cv2.putText(
            frame,
            f"Signal: {signal_state}",
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # Show video
        cv2.imshow(
            "Intelligent Traffic Management System",
            frame
        )

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()

    return average_vehicle_count