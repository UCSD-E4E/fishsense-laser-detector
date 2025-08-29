import os
from PIL import Image

def main():
    folder = "dataset/data"

    count_vertical = 0
    count_total = 0

    for filename in os.listdir(folder):
        if filename.lower().endswith(".jpg"):
            path = os.path.join(folder, filename)
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    if height > width:
                        count_vertical += 1
                    count_total += 1
                    if count_total % 100 == 0:
                        print(count_total)
            except Exception as e:
                print(f"Skipping {filename}: {e}")

    print(f"Total JPG images: {count_total}")
    print(f"Vertical images: {count_vertical}")
    print(f"Horizontal images: {count_total - count_vertical}")

if __name__ == '__main__':
    main()
