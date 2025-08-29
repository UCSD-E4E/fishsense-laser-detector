import os

txt_file = "dataset/splits/fish/train.txt"
directory = "dataset/data/"
filetype = ".JPG"

with open(txt_file, "r") as f:
    lines = f.read().splitlines()

missing = []
existing = []

for line in lines:
    img_md5, x, y = line.split(",")
    full_path = os.path.join(directory, img_md5 + filetype)

    if os.path.exists(full_path):
        existing.append(full_path)
    else:
        missing.append(full_path)

print(f"Total entries: {len(lines)}")
print(f"Found images: {len(existing)}")
print(f"Missing images: {len(missing)}")

# Write missing images to a file
with open("missing_images.txt", "w") as f:
    for m in missing:
        f.write(m + "\n")