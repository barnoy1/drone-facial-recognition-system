import os
import requests
import time
from PIL import Image
from io import BytesIO

def download_face(name):
    try:
        # Add a timestamp to avoid caching
        url = f"https://thispersondoesnotexist.com?{int(time.time())}"
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            save_path = os.path.join("../../reference_images", f"{name}.jpg")
            img.save(save_path)
            print(f"Successfully downloaded {name}.jpg")
            # Sleep to ensure we get different images
            time.sleep(1)
        else:
            print(f"Failed to download {name}.jpg")
    except Exception as e:
        print(f"Error downloading {name}.jpg: {str(e)}")

def main():
    # Create reference_images directory if it doesn't exist
    os.makedirs("../../reference_images", exist_ok=True)
    
    # Download sample faces with different names
    sample_names = ["person1", "person2", "person3"]
    
    print("Downloading sample face images...")
    for name in sample_names:
        download_face(name)
    print("Download complete!")

if __name__ == "__main__":
    main()