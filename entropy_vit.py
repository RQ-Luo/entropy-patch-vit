from PIL import Image, ImageDraw
import numpy as np

def calculate_entropy(img_array):
    entropy_r = calculate_channel_entropy(img_array[:, :, 0])
    entropy_g = calculate_channel_entropy(img_array[:, :, 1])
    entropy_b = calculate_channel_entropy(img_array[:, :, 2])
    average_entropy = (entropy_r + entropy_g + entropy_b) / 3.0
    return average_entropy

def calculate_channel_entropy(channel):
    flat_channel = channel.flatten()
    histogram, _ = np.histogram(flat_channel, bins=256, range=[0, 256])
    probability_distribution = histogram / flat_channel.size
    entropy = -np.sum(probability_distribution * np.log2(probability_distribution + 1e-10))
    return entropy

def iterative_split(image):
    width, height = image.size
    result_list = []
    out = image.copy()
    draw = ImageDraw.Draw(out)

    stack = [(0, 0, width, height)]

    while stack:
        x, y, width, height = stack.pop()
        patch = image.crop((x, y, x + width, y + height))
        mid_x = x + width // 2
        mid_y = y + height // 2
        if width * height <= 300 or calculate_entropy(np.array(patch)) < 5.5:
            draw.rectangle([x, y, x + width, y + height], outline="white")
            result_list.append(patch)
            #draw.text((mid_x-5, mid_y-5), str(len(result_list)), fill="red", align="center")
        else:
            stack.append((x, mid_y, mid_x - x, y + height - mid_y))
            stack.append((mid_x, mid_y, x + width - mid_x, y + height - mid_y))
            stack.append((mid_x, y, x + width - mid_x, mid_y - y))
            stack.append((x, y, mid_x - x, mid_y - y))

    return result_list, out

image = Image.open("image.jpg")
transform = Compose([Resize((512, 512))])
x = transform(image)
result_patches, result_image = iterative_split(x)
result_image