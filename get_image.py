from skimage import io
import os

def download_and_save_image(url, save_folder, filename=None):
    # Download image from URL
    image = io.imread(url)
    
    # Ensure save folder exists
    os.makedirs(save_folder, exist_ok=True)
    
    # Determine filename
    if filename is None:
        filename = url.split('/')[-1]
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            filename += '.png'  # Default extension
    
    save_path = os.path.join(save_folder, filename)
    
    # Save image
    io.imsave(save_path, image)
    print(f"Image saved to {save_path}")

# Example usage:
url = 'https://vincmazet.github.io/image-labs/_downloads/6686990507c201e82d66d24abf45d167/chess.png'
download_and_save_image(url, 'TP6')