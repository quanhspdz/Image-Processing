from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np

# Load a pretrained YOLOv8n model
model = YOLO('best.pt')

# Load the image
image_path = 'Inference/test.jpg'
image = cv2.imread(image_path)

# Run inference
results = model(image_path)  # results list

# Process the results
for r in results:
    # Access the list of boxes (predictions)
    boxes = r.boxes

    # Loop through each box and draw the bounding box on the image
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates (xmin, ymin, xmax, ymax)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Draw the rectangle

# Convert the BGR image to RGB for Pillow
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to PIL Image object
im = Image.fromarray(image_rgb)

# Display the image
im.show()

# Save the image with results
im.save('Inference/results.jpg')
