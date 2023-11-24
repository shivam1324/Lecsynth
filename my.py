import cv2
import easyocr
import subprocess

# Function to extract text from an image using easyocr
def extract_text(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image, detail=0)
    return result

video_path = 'sam.mp4'
cap = cv2.VideoCapture(video_path)
all_text = "" 

if not cap.isOpened():
    print("Error opening video file")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    text = extract_text(frame)
    all_text += " ".join(text) + " "
cap.release()
cv2.destroyAllWindows()
print("Combined Sentence: ", all_text)