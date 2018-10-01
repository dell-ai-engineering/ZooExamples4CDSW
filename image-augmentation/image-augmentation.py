from zoo.common.nncontext import init_nncontext
from zoo.feature.image import *
import cv2
import numpy as np
from IPython.display import Image, display
sc = init_nncontext("Image Augmentation Example")

# create LocalImageSet from an image
local_image_set = ImageSet.read("/home/cdsw/image-augmentation/image/test.jpg")

# create LocalImageSet from an image folder
local_image_set = ImageSet.read("/home/cdsw/image-augmentation/image")

# create LocalImageSet from list of images
image = cv2.imread("/home/cdsw/image-augmentation/image/test.jpg")
local_image_set = LocalImageSet([image])

print(local_image_set.get_image())
print('isDistributed: ', local_image_set.is_distributed(), ', isLocal: ', local_image_set.is_local())

# create DistributedImageSet from an image
distributed_image_set = ImageSet.read("/home/cdsw/image-augmentation/image/test.jpg", sc, 2)

# create DistributedImageSet from an image folder
distributed_image_set = ImageSet.read("/home/cdsw/image-augmentation/image/", sc, 2)

# create LocalImageSet from image rdd
image = cv2.imread("/home/cdsw/image-augmentation/image/test.jpg")
image_rdd = sc.parallelize([image], 2)
label_rdd = sc.parallelize([np.array([1.0])], 2)
distributed_image_set = DistributedImageSet(image_rdd, label_rdd)

images_rdd = distributed_image_set.get_image()
label_rdd = distributed_image_set.get_label()
print(images_rdd)
print(label_rdd)
print('isDistributed: ', distributed_image_set.is_distributed(), ', isLocal: ', distributed_image_set.is_local())
print('total images:', images_rdd.count())

# Transform images
path = "/home/cdsw/image-augmentation/image/test.jpg"
    
def transform_display(transformer, image_set):
    out = transformer(image_set)
    cv2.imwrite('/home/cdsw/image-augmentation/tmp/tmp.jpg', out.get_image(to_chw=False)[0])
    display(Image(filename='/home/cdsw/image-augmentation/tmp/tmp.jpg'))
    
# Adjust the image brightness
brightness = ImageBrightness(0.0, 32.0)
image_set = ImageSet.read(path)
transform_display(brightness, image_set)

# Adjust image hue
transformer = ImageHue(-18.0, 18.0)
image_set = ImageSet.read(path)
transform_display(transformer, image_set)

# Adjust image saturation
transformer = ImageSaturation(10.0, 20.0)
image_set= ImageSet.read(path)
transform_display(transformer, image_set)

# Random change the channel of an image
transformer = ImageChannelOrder()
image_set = ImageSet.read(path)
transform_display(transformer, image_set)

# Random adjust brightness, contrast, hue, saturation
transformer = ImageColorJitter()
image_set = ImageSet.read(path)
transform_display(transformer, image_set)

# Resize the roi(region of interest) according to scale
transformer = ImageResize(300, 300)
image_set = ImageSet.read(path)
transform_display(transformer, image_set)

# Resize the image, keep the aspect ratio. scale according to the short edge
transformer = ImageAspectScale(200, max_size = 3000)
image_set = ImageSet.read(path)
transform_display(transformer, image_set)

# Resize the image by randomly choosing a scale
transformer = ImageRandomAspectScale([100, 300], max_size = 3000)
image_set = ImageSet.read(path)
transform_display(transformer, image_set)

# Image channel normalize
transformer = ImageChannelNormalize(20.0, 30.0, 40.0, 2.0, 3.0, 4.0)
image_set = ImageSet.read(path)
transform_display(transformer, image_set)

# Pixel level normalizer, data(Pixel) = data(Pixel) - mean(Pixels)
%%time
print("PixelNormalize takes nearly one and a half minutes. Please wait a moment.")
means = [2.0] * 3 * 500 * 375
transformer = ImagePixelNormalize(means)
image_set = ImageSet.read(path)
transform_display(transformer, image_set)

# Crop a cropWidth x cropHeight patch from center of image.
transformer = ImageCenterCrop(200, 200)
image_set = ImageSet.read(path)
transform_display(transformer, image_set)

# Random crop a cropWidth x cropHeight patch from an image.
transformer = ImageRandomCrop(200, 200)
image_set = ImageSet.read(path)
transform_display(transformer, image_set)

# Crop a fixed area of image
transformer = ImageFixedCrop(0.0, 0.0, 200.0, 200.0, False)
image_set = ImageSet.read(path)
transform_display(transformer, image_set)

# Fill part of image with certain pixel value
transformer = ImageFiller(0.0, 0.0, 0.5, 0.5, 255)
image_set = ImageSet.read(path)
transform_display(transformer, image_set)

# Expand image, fill the blank part with the meanR, meanG, meanB
transformer = ImageExpand(means_r=123, means_g=117, means_b=104,
                        max_expand_ratio=2.0)
image_set = ImageSet.read(path)
transform_display(transformer, image_set)

# Flip the image horizontally
transformer = ImageHFlip()
image_set = ImageSet.read(path)
transform_display(transformer, image_set)
