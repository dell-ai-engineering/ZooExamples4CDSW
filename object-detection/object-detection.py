# import necesary libraries
import os
from IPython.display import Image, display
%pylab inline
from moviepy.editor import *

# import necessary modules
from zoo.common.nncontext import *
from zoo.feature.image import *
from zoo.models.image.objectdetection import *

# init SparkContext
sc = init_nncontext("Object Detection Example")

# Load pretrained Analytics Zoo model. Here we use a SSD-MobileNet pretrained by PASCAL VOC dataset.
try:
    model_path = "hdfs:///user/leelau/zoo/obj-det/apps/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model"
    model = ObjectDetector.load_model(model_path)
    print("load model done")
except Exception as e:
    print("The pretrain model doesn't exist")
    print("you can run $ANALYTICS_ZOO_HOME/apps/object-detection/download_model.sh to download the pretrain model")
    

# Load the video and get a short clip. Take this clip as a sequence of frames by given fps.
try:
    path = '/home/cdsw/object-detection/apps/train_dog.mp4'
    myclip = VideoFileClip(path).subclip(8,18)
except Exception as e:
    print("The video doesn't exist.")
    print("Please prepare the video first.")

video_rdd = sc.parallelize(myclip.iter_frames(fps=5))
image_set = DistributedImageSet(video_rdd)


# Predict and visualize detection back to clips
# Having prepared the model, we can start detecting objects
# Read the image as ImageSet(local/distributed) -> Perform prediction -> Visualize the detections in original images
output = model.predict_image_set(image_set)

config = model.get_config()
visualizer = Visualizer(config.label_map())
visualized = visualizer(output).get_image(to_chw=False).collect()

# Save clips to file
# Make sequence of frames back to a clip by given fps.
clip = ImageSequenceClip(visualized, fps=5)

output_path = '/home/cdsw/object-detection/out.mp4'
clip.write_videofile(output_path, audio=False)
clip.write_gif("train_dog.gif")

# Display Object Detection Video
# Display the prediction of the model.
with open("train_dog.gif",'rb') as f:
    display(Image(f.read()))
