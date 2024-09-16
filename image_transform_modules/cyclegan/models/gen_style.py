import numpy as np
import torch
from . import networks
from PIL import Image

class img2style():
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    
    def __init__(self, ckpt, device_id):
        """Initialize the pix2pix class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.netG = 'resnet_9blocks'
        self.norm = 'instance'
        self.no_dropout = True    # without dropout during inference
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.ckpt = ckpt
        self.device_id = device_id   # device_id=0/1 
        self.gpu_ids = [self.device_id]
        self.device = "cuda:{}".format(self.gpu_ids[0])
       
        # we only need Generator A(from RGB to infrared/from day to night) --> single direction
        self.netG = networks.define_G(self.input_nc, self.output_nc, self.ngf, self.netG,
                                      self.norm, not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
          
    def load_networks(self):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '***.pth' % 
        """
        state_dict = torch.load(self.ckpt, map_location=str(self.device))
        # we only need net_G_A.pth --> single direction
        self.netG = self.netG.module  # self.netG is an instance of torch.nn.DataParallel
        self.netG.load_state_dict(state_dict, strict=False)
        
    def style_transfer(self, data):
        with torch.no_grad():
            image = data['image']
            style_img_numpy = self.netG(image).cpu().numpy()
            
        return style_img_numpy[0]
    
    # save style_img
    def save_style_img(self, input_image, save_path, imtype=np.uint8):
        """
        style_img: np.ndarray, H * W
        """
        image_numpy = (np.transpose(input_image, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling    
        image_numpy = image_numpy.astype(imtype) 
        
        Image.fromarray(image_numpy).save(save_path)