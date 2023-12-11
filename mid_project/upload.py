import tkinter as tk
from tkinter import filedialog,messagebox
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import warnings
import numpy as np
import argparse
import cv2
import numpy as np
import torch
from torchvision import models
import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


warnings.filterwarnings('ignore')



def upload_file():
    global selectFile

    selectFile = tk.filedialog.askopenfilename()
    entry1.insert(0, selectFile)
    
    btn2 = tk.Button(frm2, text='Start', command=show)
    btn2.grid(row=0, column=0, ipadx='3', ipady='3', padx='10', pady='20')

    

def predict_file():
    global img
    global photo
    global selectFile
#     def get_args():
#         parser = argparse.ArgumentParser()
#         parser.add_argument('--use-cuda', action='store_true', default=False,
#                             help='Use NVIDIA GPU acceleration')
#         parser.add_argument(
#             '--image-path',
#             type=str,
#             default=selectFile,
#             help='Input image path')
#         parser.add_argument('--aug_smooth', action='store_true',
#                             help='Apply test time augmentation to smooth the CAM')
#         parser.add_argument(
#             '--eigen_smooth',
#             action='store_true',
#             help='Reduce noise by taking the first principle componenet'
#             'of cam_weights*activations')
#         parser.add_argument('--method', type=str, default='gradcam',
#                             choices=['gradcam', 'gradcam++',
#                                     'scorecam', 'xgradcam',
#                                     'ablationcam', 'eigencam',
#                                     'eigengradcam', 'layercam', 'fullgrad'],
#                             help='Can be gradcam/gradcam++/scorecam/xgradcam'
#                                 '/ablationcam/eigencam/eigengradcam/layercam')

#         args = parser.parse_args()
#         args.use_cuda = args.use_cuda and torch.cuda.is_available()
#         if args.use_cuda:
#             print('Using GPU for acceleration')
#         else:  
#             print('Using CPU for computation')
#         return args
#     args = get_args()
    train_transform = A.Compose([
        A.Resize(128,128),
        A.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24]),
        ToTensorV2()
])
    img=cv2.cvtColor(cv2.imread(selectFile), cv2.COLOR_BGR2RGB)
    img=train_transform(image=img)['image']
    img=img.view(1,3,128,128)

    net = torch.load('net.pkl')  # 加载文件net.pkl, 将其内容赋值给output
    net.eval()
#     model = net
#     target_layers = [model.layer4]
#     rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
#     rgb_img = np.float32(rgb_img) / 255
#     input_tensor = preprocess_image(rgb_img,
#                                     mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
#     targets = None
#     cam_algorithm = methods[args.method]
#     with cam_algorithm(model=model,
#                        target_layers=target_layers,
#                        use_cuda=args.use_cuda) as cam:


#         cam.batch_size = 32
#         grayscale_cam = cam(input_tensor=input_tensor,
#                             targets=targets,
#                             aug_smooth=args.aug_smooth,
#                             eigen_smooth=args.eigen_smooth)

#         # Here grayscale_cam has only one image in the batch
#         grayscale_cam = grayscale_cam[0, :]

#         cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

#         # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
#         cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

#     gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
#     gb = gb_model(input_tensor, target_category=None)

#     cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
#     cam_gb = deprocess_image(cam_mask * gb)

    
    outputs = net(img)#.reshape(1, 3)

    p = nn.Softmax(dim=1)(outputs)#.unsqueeze(dim=1)
    predicted = torch.argmax(outputs)



    labels_dict=['benign','malignant','normal']
    tk.messagebox.askokcancel("Feedback","The prediction of the image is {}, with a probability of {:.3f}".format(labels_dict[predicted],float(p[0][predicted])))
def show():
    global selectFile
    global photo1,photo2
    global img
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--use-cuda', action='store_true', default=False,
                            help='Use NVIDIA GPU acceleration')
        parser.add_argument(
            '--image-path',
            type=str,
            default=selectFile,
            help='Input image path')
        parser.add_argument('--aug_smooth', action='store_true',
                            help='Apply test time augmentation to smooth the CAM')
        parser.add_argument(
            '--eigen_smooth',
            action='store_true',
            help='Reduce noise by taking the first principle componenet'
            'of cam_weights*activations')
        parser.add_argument('--method', type=str, default='gradcam',
                            choices=['gradcam', 'gradcam++',
                                    'scorecam', 'xgradcam',
                                    'ablationcam', 'eigencam',
                                    'eigengradcam', 'layercam', 'fullgrad'],
                            help='Can be gradcam/gradcam++/scorecam/xgradcam'
                                '/ablationcam/eigencam/eigengradcam/layercam')

        args = parser.parse_args()
        args.use_cuda = args.use_cuda and torch.cuda.is_available()
        if args.use_cuda:
            print('Using GPU for acceleration')
        else:
            print('Using CPU for computation')

        return args

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    model = torch.load('net.pkl')  # 加载文件net.pkl, 将其内容赋值给output
    model.eval()

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    target_layers = [model.layer4]
    train_transform = A.Compose([
    A.Resize(128,128),
    A.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24]),
    ToTensorV2()
])
    # img=cv2.cvtColor(cv2.imread(args.image_path), cv2.COLOR_BGR2RGB)
    # img=train_transform(image=img)['image']
    # img=img.view(1,3,128,128)

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])


    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)
    cv2.imwrite(f'{args.method}_gb.jpg', gb)

    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)

    cv2.imwrite(f'{args.method}_cam_gb.png', cam_gb)

    img2 = Image.open('gradcam_cam.jpg')  # 打开图片
    photo1 = ImageTk.PhotoImage(img2)  # 用PIL模块的PhotoImage打开
    imglabel = tk.Label(root, image=photo1)
    imglabel.grid(row=0, column=0, columnspan=3)
    img = Image.open(selectFile)  # 打开图片
    photo2 = ImageTk.PhotoImage(img)  # 用PIL模块的PhotoImage打开
    imglabel2 = tk.Label(root, image=photo2)
    imglabel2.grid(row=0, column=3, columnspan=3)
    predict_file()

    
    # imglabel2.grid(row=0, column=0, columnspan=3)








methods = \
    {"gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad}
selectFile=None
root = tk.Tk()
# root.geometry('400x300')
root.title = 'The machine of diagnosis for breast cancel'
frm = tk.Frame(root)
frm.grid(padx='20', pady='30')
frm2 = tk.Frame(root)
frm2.grid(padx='100', pady='5')
btn = tk.Button(frm, text='Upload file', command=upload_file)
btn.pack()
btn.grid(row=0, column=0, ipadx='3', ipady='3', padx='10', pady='20')
entry1 = tk.Entry(frm, width='40')
entry1.grid(row=0, column=1)

root.mainloop()

    

    