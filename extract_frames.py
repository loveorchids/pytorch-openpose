import os, glob

root = os.path.expanduser("~/Pictures/dataset/reid/20191119_FaceAlignment/FA_4")

for i, video in enumerate(sorted(glob.glob(root + "/*_raw.mp4"))):
    name = video[video.rfind("/")+1:-8]
    angle = name[name.find("_")+1:name.rfind("_")]
    output_dir = os.path.join(root, name)
    output_dir = os.path.expanduser("~/Documents/pytorch-openpose/images")
    #print("Video Name: %s"%name)
    #print("Person angle: %s"%angle)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    cmd = 'ffmpeg -i {} -vf "fps=4" {}/%04d_{}.jpg'.format(video, output_dir, angle)
    print(cmd)
    #print("-------------------------")
    #print("")
