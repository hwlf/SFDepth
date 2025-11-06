import random
from copy import deepcopy

import PIL.Image as pil
import skimage.transform
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from datasets.kitti_utils import *
from utils.seg_utils import labels,decode_seg_map
import cv2

import matplotlib.pyplot as plt


def canny_edge_detector(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Calculate gradients using Sobel operators
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    # Calculate gradient magnitude and direction
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_magnitude = torch.unsqueeze(torch.tensor(gradient_magnitude), 0)  # value: between [-pi, pi]
    return gradient_magnitude

def canny_edge_detector1(image):
    # Convert the image to grayscale
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    # Calculate gradients using Sobel operators
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    # Calculate gradient magnitude and direction
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_magnitude = torch.unsqueeze(torch.tensor(gradient_magnitude), 0)  # value: between [-pi, pi]
    return gradient_magnitude



def pil_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    if mode == 'P':
        return Image.open(path)
    else:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class KittiDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """

    def __init__(self,

                 height,
                 width,
                 frame_idxs,
                 filenames,
                 data_path,
                 num_scales,
                 is_train,
                 img_ext='.png',
                 ):
        super(KittiDataset, self).__init__()



        self.height = height
        self.width = width
        self.frame_idxs = frame_idxs
        self.filenames = filenames
        self.data_path = data_path
        self.num_scales = num_scales
        self.is_train = is_train
        self.img_ext = img_ext
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()


        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.resize_img = {}
        self.resize_seg = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize_img[i] = transforms.Resize((self.height // s, self.width // s),
                                                   interpolation=Image.ANTIALIAS)

            self.resize_seg[i] = transforms.Resize((self.height // s, self.width // s),
                                                   interpolation=Image.BILINEAR)

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=Image.ANTIALIAS)
        if is_train:
            self.load_depth = False

        else:
            self.load_depth = self.check_depth()

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

        # self.class_dict = self.get_classes(self.filenames)

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose seg_networks receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, _ = k
                inputs[n[0] + '_size'] = torch.tensor(inputs[(n, im, -1)].size)
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize_img[i](inputs[(n, im, -1)])

                del inputs[(n, im, -1)]

            if "seg" in k:
                n, im, _ = k
                inputs[n[0] + '_size'] = torch.tensor(inputs[(n, im, -1)].size)
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize_seg[i](inputs[(n, im, -1)])

                del inputs[(n, im, -1)]

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            elif "seg" in k:
                n, im, i = k
                inputs[(n, im, i)] = torch.tensor(np.array(f)).float().unsqueeze(0)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) in [3, 4, 2]:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None
        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        if side is None:
            if do_color_aug:
                color_aug = transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)
            else:
                color_aug = (lambda x: x)
            self.preprocess(inputs, color_aug)
            return inputs

        inputs[("seg", 0, -1)] = self.get_seg_map(folder, frame_index, side, do_flip)

        # 复制原始索引图（确保不修改 `inputs[("seg", 0, -1)]`）
        seg_for_gradient = np.array(inputs[("seg", 0, -1)])

        # 合并类别：将 sidewalk (1) 和 terrain (9) 变为 road (0)
        #seg_for_gradient[(seg_for_gradient == 1) | (seg_for_gradient == 9)] = 0

        road_mask = (seg_for_gradient == 0).astype(np.uint8)

        road_mask_resized = self.resize[0](Image.fromarray(road_mask * 255))

        inputs["road_mask"] = torch.tensor(np.array(road_mask_resized) / 255.0, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        # # 检查尺寸
        # print("road_mask 尺寸:", inputs["road_mask"].shape)  # 应该输出 (1, H, W)
        #
        # # 可视化 road_mask
        # def visualize_mask(mask, title="Road Mask"):
        #     """
        #     可视化二值掩码。
        #     :param mask: 二值掩码，形状为 (1, H, W) 或 (H, W)
        #     :param title: 图像标题
        #     """
        #     if isinstance(mask, torch.Tensor):
        #         mask = mask.squeeze(0).numpy()  # 去掉批次维度并转换为 NumPy 数组
        #     plt.imshow(mask, cmap='gray')
        #     plt.title(title)
        #     plt.axis('off')  # 不显示坐标轴
        #     plt.show()
        #
        # # 可视化 road_mask
        # visualize_mask(inputs["road_mask"])
        #_______________________________________________________________
        #可视化
        # 生成可视化 mask
        # road_mask_vis = (road_mask * 255).astype(np.uint8)
        #
        # # 保存可视化结果
        # cv2.imwrite("road_mask.png", road_mask_vis)
        #
        # # 或者转换为 PIL 图像
        # road_mask_pil = Image.fromarray(road_mask_vis)
        # road_mask_pil.show()  # 显示图片
        # plt.figure(figsize=(18, 6))
        #
        # # 显示原始分割图
        # plt.subplot(2, 3, 1)
        # plt.imshow(inputs[("seg", 0, -1)], cmap='gray')  # 使用灰度颜色映射
        # plt.title("Original Segmentation Map")
        # plt.axis('off')  # 关闭坐标轴
        #
        # # 显示原始分割图的梯度
        # plt.subplot(2, 3, 4)
        # plt.imshow(magnitude1.squeeze(), cmap='gray')  # 去掉 batch 维度并显示
        # plt.title("Gradient of Original Segmentation Map")
        # plt.axis('off')
        #
        # # 显示上色后的分割图
        # plt.subplot(2, 3, 2)
        # plt.imshow(colored_seg_pil)
        # plt.title("Colored Segmentation Map")
        # plt.axis('off')
        #
        # # 显示上色后的分割图的梯度
        # plt.subplot(2, 3, 5)
        # plt.imshow(magnitude.squeeze(), cmap='gray')  # 去掉 batch 维度并显示
        # plt.title("Gradient of Colored Segmentation Map")
        # plt.axis('off')
        #
        # # 显示图像
        # plt.tight_layout()
        # plt.show()


        K = deepcopy(self.K)
        if do_flip:
            K[0, 2] = 1 - K[0, 2]

        K[0, :3] *= self.width
        K[1, :3] *= self.height
        inv_K = np.linalg.pinv(K)
        inputs[("K")] = torch.from_numpy(K)
        inputs[("inv_K")] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])
        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        im_path = self.get_image_path(folder, frame_index, side)

        color = self.loader(im_path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        if side is not None:
            image_path = os.path.join(
                self.data_path,
                folder,
                "image_0{}/data".format(self.side_map[side]),
                f_str)
        else:
            image_path = os.path.join(
                self.data_path, folder, "{}".format(f_str))
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_seg_map(self, folder, frame_index, side, do_flip):
        path = self.get_image_path(folder, frame_index, side)
        path = path.replace('KITTI', 'KITTI/segmentation')
        path = path.replace('/data', '')

        seg = self.loader(path, mode='P')
        seg_copy = np.array(seg.copy())

        for k in np.unique(seg):
            seg_copy[seg_copy == k] = labels[k].trainId
        seg = Image.fromarray(seg_copy, mode='P')

        if do_flip:
            seg = seg.transpose(pil.FLIP_LEFT_RIGHT)
        return seg
