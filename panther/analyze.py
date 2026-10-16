from PIL import Image
img = Image.open('panther/1.PNG')
width, height = img.size
print(f'Image size: {width}x{height}')

# Check if there are non-gray pixels
non_gray = 0
for x in range(width):
    for y in range(height):
        r, g, b, a = img.getpixel((x, y))
        if r != g or g != b:
            non_gray += 1
print(f'Non-gray pixels: {non_gray} out of {width*height}')

# Get unique colors sample
colors = set()
for x in range(0, width, 10):
    for y in range(0, height, 10):
        colors.add(img.getpixel((x,y)))
print(f'Sample unique colors (every 10th pixel): {len(colors)} unique')
for c in sorted(colors)[:30]:
    print(f'  {c}')