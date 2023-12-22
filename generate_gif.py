from PIL import Image
import os

def generate_gif(folder_path):
    # Retrieve all image files in the folder
    images = [img for img in os.listdir(folder_path) if img.endswith(".png")]

    # Sort the files to ensure correct order in the GIF
    images.sort()

    # Create an empty list to store the image objects
    frames = []

    # Open each image and append it to the frames list
    for i in images:
        new_frame = Image.open(os.path.join(folder_path, i))
        frames.append(new_frame)

    # Create the GIF
    frames[0].save(os.path.join(folder_path, "output.gif"), format='GIF',
                   append_images=frames[1:], save_all=True, duration=300, loop=0)

    print('finished')
    return os.path.join(folder_path, "output.gif")

# Example usage
gif_path = generate_gif("truth")
# print("GIF created at:", gif_path)
