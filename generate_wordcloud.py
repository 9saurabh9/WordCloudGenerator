import numpy as np
import cv2
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image

def preprocess_mask(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Ensure the image has an alpha channel (RGBA)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    # Extract alpha channel
    alpha = image[:, :, 3] if image.shape[2] == 4 else None

    # Convert to grayscale
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)

    # Create an RGB image (initialize with white)
    rgb_result = np.ones((gray.shape[0], gray.shape[1], 3), dtype=np.uint8) * 255  # Start as white

    # Black threshold (increase to include more lighter areas)
    black_threshold = 90
    rgb_result[gray <= black_threshold] = [0, 0, 0]  # Set dark areas to black

    # Replace transparent areas with white
    if alpha is not None:
        rgb_result[alpha == 0] = [255, 255, 255]

    return rgb_result

def create_word_cloud(image_path, text_path, output_path="word_cloud_portrait.png"):
    # Preprocess the mask to extract face outline
    mask = preprocess_mask(image_path)

    # Read text file
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Generate Word Cloud
    wordcloud = WordCloud(width=800, height=1000, max_words=500,
                          background_color="white", mask=mask,
                          prefer_horizontal=0.8).generate(text)

    # Convert to image
    wordcloud_image = wordcloud.to_array()
    result = Image.fromarray(wordcloud_image)

    # Save and display
    result.save(output_path)
    plt.figure(figsize=(10, 12))
    plt.imshow(result, cmap="gray")
    plt.axis("off")
    plt.show()

# Run the function
create_word_cloud("face.png", "alice.txt", "word_cloud_portrait.png")
