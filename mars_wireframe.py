from lib.pare.models.hmr import HMR
from PIL import Image
from torchvision.transforms import transforms
from lib.pare.utils.vibe_renderer import Renderer

import cv2
import numpy as np

transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    # transforms.RandomRotation(degrees=45)
])


h = HMR(pretrained="data/pare/checkpoints/spin_model_checkpoint.pth.tar")
fName = "/home/akunchala/Documents/z_Datasets/MARS_Dataset/bbox_train/0379/0379C1T0001F014.jpg"
img_orig = cv2.cvtColor(cv2.imread(fName), cv2.COLOR_BGR2RGB)
img = transform_eval(img_orig).unsqueeze(dim=0)
x = h(img)
print(x.keys())

renderer = Renderer()
cam = x["pred_cam"].detach().numpy().squeeze()

img_render = cv2.resize(cv2.imread(fName),(224,224))

sx = cam[0] * (1. / (128 / 224))
sy = cam[0] * (1. / (256 / 224))


img_ = renderer.render(
    img_render,
    x["smpl_vertices"].detach().numpy().squeeze(),
    cam=[sx,sy,cam[1],cam[2]],
    color=(0.5, 0.9108974590556427, 1.0),
)

cv2.imwrite("out.png", cv2.resize(img_,(128,256)))
