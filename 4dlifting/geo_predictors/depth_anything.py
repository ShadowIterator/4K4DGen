import torch
from torchvision.transforms import Compose
import cv2
import torch.nn.functional as F
from Depth_Anything.depth_anything.dpt import DepthAnything
from Depth_Anything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


class DepthAnythingPredictor():
    def __init__(self, scale):
        # pass
        self.model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format('vitl')).eval()
        self.transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        self.scale = scale
        self.min_depth = 0
        self.max_depth = 2.5
        # self.depth_anything = DepthAnything.from_pretrained('/home/tiger/.cache/huggingface/hub/models--LiheYoung--depth_anything_vitl14').to('cuda').eval()
        # super().__init__()
        # self.img_size = 512 ### 384 sz: try 512
        # ckpt_path = 'pre_checkpoints/omnidata_dpt_depth_v2.ckpt'
        # self.model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=1)
        # self.model.to(torch.device('cpu'))
        # checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        # if 'state_dict' in checkpoint:
        #     state_dict = {}
        #     for k, v in checkpoint['state_dict'].items():
        #         state_dict[k[6:]] = v
        # else:
        #     state_dict = checkpoint

        # self.model.load_state_dict(state_dict)
        # self.trans_totensor = transforms.Compose([transforms.Resize(self.img_size, interpolation=Image.BILINEAR),
        #                                           transforms.CenterCrop(self.img_size),
        #                                           transforms.Normalize(mean=0.5, std=0.5)])
        # # self.trans_rgb  = transforms.Compose([transforms.Resize(512, interpolation=PIL.Image.BILINEAR),
        # #                   transforms.CenterCrop(512)])

    def predict_disparity(self, image, **kwargs):
        self.model.to(torch.device('cuda'))
        image = image[0].permute(1,2,0).detach().cpu().numpy()
        h, w = image.shape[:2]
        image = self.transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to('cuda')
        disp = self.model(image)
        disp = F.interpolate(disp[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

        # pass
        # self.model.to(torch.device('cuda'))
        # img_tensor = self.trans_totensor(img)
        # output = self.model(img_tensor).clip(0., 1.)
        self.model.to(torch.device('cpu'))
        return disp
        # # output = F.interpolate(output[:, None], size=(512, 512), mode='bicubic')
        # output = output.clip(0., 1.)
        # # output = 1. - output
        # output = 1. / (output + 1e-6)
        # return output[:, None]

    def predict_depth(self, img, **kwargs):
        disp = self.predict_disparity(img, **kwargs)
        disp = disp[None, None, ...]
        depth = self.scale / (disp + 1e-5)
        depth_clip = depth.clip(self.min_depth, self.max_depth)
        return depth_clip
        # pass
        # self.model.to(torch.device('cuda'))
        # img_tensor = self.trans_totensor(img)
        # output = self.model(img_tensor).clip(0., 1.)
        # self.model.to(torch.device('cpu'))
        # # output = F.interpolate(output[:, None], size=(512, 512), mode='bicubic')
        # output = output.clip(0., 1.)
        # return output[:, None]

