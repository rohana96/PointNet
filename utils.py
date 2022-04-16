import os
import torch
import pytorch3d
from PIL import Image, ImageDraw, ImageSequence, ImageFont
import io
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
import imageio
import numpy as np

def save_checkpoint(epoch, model, args, best=False):
    if best:
        path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else:
        path = os.path.join(args.checkpoint_dir, 'model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def topk(accuracy, k=10, largest=True):
    """
    Args:
        accuracy(Tensor): tensor of accuracies
        k(int): no of accuracies to return
        largest(bool): if True largest k accuracies are returned
    Returns:
        topk accuracies and their indices
    """
    return torch.topk(accuracy, k=k, largest=largest, sorted=True)

def viz_seg(verts, pred_labels, target_labels, scores, path, device, num_points, limit_render=False, max_render=3):
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)
    colors = [[1.0,1.0,1.0], [1.0,0.0,1.0], [0.0,1.0,1.0],[1.0,1.0,0.0],[0.0,0.0,1.0], [1.0,0.0,0.0]]

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    if limit_render == True:
        max_render = min(len(pred_labels), max_render)
    else:
        max_render = len(pred_labels)
        
    # generate gifs for topk results
    for j in range(max_render):
        sample_verts = verts[j].unsqueeze(0).repeat(30,1,1).to(torch.float)
        sample_labels = pred_labels[j].unsqueeze(0)
        sample_colors = torch.zeros((1,num_points,3))

        # Colorize points based on segmentation labels
        for i in range(6):
            sample_colors[sample_labels==i, :] = torch.tensor(colors[i])

        sample_colors = sample_colors.repeat(30,1,1).to(torch.float)

        point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

        renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
        rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)

        gif_path = f'{path}{j}.gif'
        imageio.mimsave(gif_path, rend, fps=15)

        labelled_gif_seg(gif_path, scores[j])

def viz_cls(verts, pred_labels, target_labels, scores, path, device, num_points, limit_render=False, max_render=3):
    """
    visualize classification result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)
    colors = torch.tensor([[1.0,1.0,0.0]])

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    if limit_render == True:
        max_render = min(len(pred_labels), max_render)
    else:
        max_render = len(pred_labels)
    # generate gifs for topk results
    for i in range(max_render):
        sample_verts = verts[i].unsqueeze(0).repeat(30,1,1).to(torch.float)
        sample_labels = pred_labels.unsqueeze(0)
        sample_colors = colors.unsqueeze(0)
        sample_colors = colors.repeat(30,num_points,1).to(torch.float)

        point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

        renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
        rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)
        
        pred_ind = int(pred_labels[i])
        target_ind = int(target_labels[i])

        gif_path = f'{path}_{class_dict[target_ind]}_{i}.gif'
        imageio.mimsave(gif_path, rend, fps=15)

        labelled_gif(gif_path, class_dict[pred_ind], class_dict[target_ind], scores[i])


def labelled_gif(path, pred_label, target_label, score):
    """Reference: https://github.com/python-pillow/Pillow/issues/3128"""
    im = Image.open(path)
    frames = []
    for frame in ImageSequence.Iterator(im):
        d = ImageDraw.Draw(frame)
        d.text((10,10), f"pred: {pred_label}")
        d.text((10,25), f"target: {target_label}")
        d.text((10,40), f"confidence: {score*100:0.2f}%")
        del d
        b = io.BytesIO()
        frame.save(b, format="GIF")
        frame = Image.open(b)
        frames.append(frame)
    frames[0].save(path, save_all=True, append_images=frames[1:])


def labelled_gif_seg(path, score):
    """Reference: https://github.com/python-pillow/Pillow/issues/3128"""
    im = Image.open(path)
    frames = []
    for frame in ImageSequence.Iterator(im):
        d = ImageDraw.Draw(frame)
        d.text((10,10), f"accuracy: {score*100:0.2f}%")
        del d
        b = io.BytesIO()
        frame.save(b, format="GIF")
        frame = Image.open(b)
        frames.append(frame)
    frames[0].save(path, save_all=True, append_images=frames[1:])

class_dict = {0: 'chair', 1:'vase', 2:'lamp'}