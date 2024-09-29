from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2
import numpy as np
import pygame

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

pygame.init()
 
HEIGHT = 450
WIDTH = 400
FPS = 60
 
clock = pygame.time.Clock()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.Font(pygame.font.get_default_font(), 36)
pygame.display.set_caption("Game")

running = True
while running:
# Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    screen.fill((255,255,255))
    # You can use `render` and then blit the text surface ...
    text_surface = font.render("Class: " + class_name[2:], True, (0, 0, 255))
    screen.blit(text_surface, (40, 250))

    # Print prediction and confidence score
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(FPS)

camera.release()
cv2.destroyAllWindows()
pygame.quit()
