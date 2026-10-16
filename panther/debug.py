from PIL import Image
img = Image.open('panther/1.PNG')
width, height = img.size
print('Image size:', width, 'x', height)

print('\nFirst row pixels (first 20):')
for x in range(20):
    print('  (' + str(x) + ',0):', img.getpixel((x,0)))

print('\nMiddle row (row ' + str(height//2) + ', first 20):')
for x in range(20):
    print('  (' + str(x) + ',' + str(height//2) + '):', img.getpixel((x,height//2)))

print('\nLast row pixels (first 20):')
for x in range(20):
    print('  (' + str(x) + ',' + str(height-1) + '):', img.getpixel((x,height-1)))

print('\nFirst column pixels (first 10):')
for y in range(10):
    print('  (0,' + str(y) + '):', img.getpixel((0,y)))

print('\nMiddle column (col ' + str(width//2) + ', first 10):')
for y in range(10):
    print('  (' + str(width//2) + ',' + str(y) + '):', img.getpixel((width//2,y)))