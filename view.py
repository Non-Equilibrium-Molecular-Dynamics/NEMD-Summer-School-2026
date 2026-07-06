#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os.path

if os.path.isfile(sys.argv[1]):
    image_path = sys.argv[1]
    image = mpimg.imread(image_path)
    plt.imshow(image)
    plt.show()
else:
    print("ERROR: Filename "+sys.argv[1]+" not found")
