import numpy as np
from sklearn.decomposition import PCA
import os
from PIL import Image



def image_to_patches(img_array, patch_size, stride=4, return_positions=False):
    """
    Load an image and divide it into patches.
    
    Parameters:
    -----------
    img_array : ndarray
        Image as a numpy array.
    patch_size : tuple of int
        Size of patches (height, width).
    stride : tuple of int, optional
        Step size between patches (height, width).
        If None, stride = patch_size (no overlap).
    return_positions : bool, optional
        If True, return the positions of each patch in the original image.
        
    Returns:
    --------
    patches : numpy.ndarray
        Array of patches with shape (n_patches, patch_height, patch_width, n_channels).
    positions : list of tuples, optional
        List of (y, x) positions of each patch in the original image.
    """
    
    # Get image dimensions
    img_height, img_width = img_array.shape[:2]
    patch_height = patch_size
    patch_width = patch_size
    
    # Set stride to patch_size by default (no overlap)
    if stride is None:
        stride = patch_size
    stride_y = stride
    stride_x = stride
    
    # Calculate number of patches
    n_patches_y = 1 + max(0, (img_height - patch_height) // stride_y)
    n_patches_x = 1 + max(0, (img_width - patch_width) // stride_x)
    
    # Handle edge cases to ensure full coverage
    # If the patches don't cover the entire image, add one more patch
    if n_patches_y * stride_y + patch_height < img_height:
        n_patches_y += 1
    if n_patches_x * stride_x + patch_width < img_width:
        n_patches_x += 1
    
    # Initialize list to store patches and positions
    patches = []
    positions = []
    
    # Extract patches
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            # Calculate patch start positions
            y_start = min(i * stride_y, img_height - patch_height)
            x_start = min(j * stride_x, img_width - patch_width)
            
            # Extract patch
            patch = img_array[y_start:y_start + patch_height, x_start:x_start + patch_width]
            patches.append(patch)
            
            if return_positions:
                positions.append((y_start, x_start))
    
    # Convert list of patches to numpy array
    patches = np.array(patches)
    
    if return_positions:
        return patches, positions, patch_height, patch_width
    else:
        return patches
    


def patches_to_image_with_positions(patches, positions, image_height, image_width):
    """
    Reconstruct an image from patches using their original positions.
    
    Parameters:
    -----------
    patches : numpy.ndarray
        Array of patches with shape (n_patches, patch_height, patch_width, n_channels)
    positions : list of tuples
        List of (y, x) positions of each patch in the original image.
    image_height, image_width : int
        Dimensions of the output image.
        
    Returns:
    --------
    reconstructed_img : numpy.ndarray
        Reconstructed image.
    """
    # Determine image shape (grayscale or color)
    if len(patches.shape) == 3:
        reconstructed_img = np.zeros((image_height, image_width), dtype=patches.dtype)
        count = np.zeros((image_height, image_width), dtype=np.float32)
        has_channels = False
    else:
        n_channels = patches.shape[3]
        reconstructed_img = np.zeros((image_height, image_width, n_channels), dtype=patches.dtype)
        count = np.zeros((image_height, image_width, n_channels), dtype=np.float32)
        has_channels = True
    
    patch_height, patch_width = patches.shape[1], patches.shape[2]
    
    # Place each patch at its recorded position
    for idx, (y_start, x_start) in enumerate(positions):
        if has_channels:
            reconstructed_img[y_start:y_start + patch_height, x_start:x_start + patch_width, :] += patches[idx]
            count[y_start:y_start + patch_height, x_start:x_start + patch_width, :] += 1
        else:
            reconstructed_img[y_start:y_start + patch_height, x_start:x_start + patch_width] += patches[idx]
            count[y_start:y_start + patch_height, x_start:x_start + patch_width] += 1
    
    # Avoid division by zero
    count[count == 0] = 1
    reconstructed_img = reconstructed_img / count
    
    return reconstructed_img






def load_images(folder_path):
    images = []
    filenames = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', 'JPEG', 'JPG', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = np.array(Image.open(img_path))
            images.append(img)
            filenames.append(filename)
            print(f"Loaded {filename} with shape {img.shape}")
    
    return images, filenames






def preprocess_images(images, start_idx, num_imgs=300, sigma=0.1):
    # normalize image values to 0-1
    orig_img_scaled = [img_array / 255.0 for img_array in images]
    print(f'Total number of original images: len(img_scaled_orig) = {len(orig_img_scaled)}')


    # get rid of gray images
    img_scaled = [] #set of colorful RGB images
    cnt = 0
    for  i in range(start_idx, len(orig_img_scaled)):
        img = orig_img_scaled[i]
        if len(img.shape) <=2: # avoid grayscale images
            pass
        else:
            cnt += 1
            img_scaled.append(img)
        if cnt >= num_imgs:
            break


    print(f'Total number of RGB images: len(img_scaled) for training = {len(img_scaled)}')



    # get noisy images
    noisy_images = []
    clean_images = []


    for img in img_scaled:
        clean_images.append(img)
        noisy_img = img + sigma * np.random.randn(*img.shape)
        
        noisy_images.append(noisy_img)


    print(f'Number of clean images: len(clean_images) = {len(clean_images)}')
    print(f'Number of noisy images: len(noisy_images) = {len(noisy_images)}')

    return orig_img_scaled, img_scaled, noisy_images, clean_images




def get_patches_from_images(clean_images, noisy_images, patch_size=8, stride=8):
    clean_patches = []
    noisy_patches = []
    clean_positions = []
    noisy_positions = []



    N_patches_per_img = []

    for i in range(len(noisy_images)):

        cur_img_clean_patches, cur_clean_positions, patch_height, patch_width = image_to_patches(clean_images[i], patch_size, stride=stride, return_positions=True)
        cur_img_noisy_patches, cur_noisy_positions, patch_height, patch_width = image_to_patches(noisy_images[i], patch_size, stride=stride, return_positions=True)


        clean_patches.append(cur_img_clean_patches)
        noisy_patches.append(cur_img_noisy_patches)

        clean_positions.append(cur_clean_positions)
        noisy_positions.append(cur_noisy_positions)

        N_patches_per_img.append(int(cur_img_clean_patches.shape[0]))



    N_patch_total = np.sum(np.array(N_patches_per_img))
    print(f"Number of total patches in noisy images: {N_patch_total}")

    return clean_patches, noisy_patches, clean_positions, noisy_positions, N_patches_per_img




def flatten_patches(clean_patches, noisy_patches):
    clean_patches_flattened = None
    noisy_patches_flattened = None

    for i in range(len(noisy_patches)):

        if noisy_patches_flattened is None:
            noisy_patches_flattened = noisy_patches[i].reshape(noisy_patches[i].shape[0], -1)
        else:
            noisy_patches_flattened = np.vstack((noisy_patches_flattened, noisy_patches[i].reshape(noisy_patches[i].shape[0], -1)))


        if clean_patches_flattened is None:
            clean_patches_flattened = clean_patches[i].reshape(clean_patches[i].shape[0], -1)
        else:
            clean_patches_flattened = np.vstack((clean_patches_flattened, clean_patches[i].reshape(clean_patches[i].shape[0], -1)))
    return clean_patches_flattened, noisy_patches_flattened