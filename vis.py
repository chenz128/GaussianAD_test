import os
os.environ['QT_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5'
offscreen = False
if os.environ.get('DISP', 'f') == 'f':
    try:
        from pyvirtualdisplay import Display
        display = Display(visible=False, size=(2560, 1440))
        display.start()
        offscreen = True
    except:
        print("Failed to start virtual display.")

try:
    from mayavi import mlab
    import mayavi
    mlab.options.offscreen = offscreen
    print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))
except:
    print("No Mayavi installation found.")

import torch, numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from pyquaternion import Quaternion
import os
from mmdet.models.task_modules.coders import BaseBBoxCoder
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid

def save_occ(
        save_dir, 
        gaussian, 
        name,
        sem=False,
        cap=2,
        dataset='nusc'
    ):
    if dataset == 'nusc':
        # voxel_size = [0.4] * 3
        # vox_origin = [-40.0, -40.0, -1.0]
        # vmin, vmax = 0, 16
        voxel_size = [0.5] * 3
        vox_origin = [-50.0, -50.0, -5.0]
        vmin, vmax = 0, 16
    elif dataset == 'kitti':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
        vmin, vmax = 1, 19
    elif dataset == 'kitti360':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
        vmin, vmax = 1, 18

    voxels = gaussian[0].cpu().to(torch.int)
    voxels[0, 0, 0] = 1
    voxels[-1, -1, -1] = 1
    if not sem:
        voxels[..., (-cap):] = 0
        for z in range(voxels.shape[-1] - cap):
            mask = (voxels > 0)[..., z]
            voxels[..., z][mask] = z + 1 
    
    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        voxels.shape, voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    if not sem:
        fov_voxels = fov_grid_coords[
            (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 100)
        ]
    else:
        if dataset == 'nusc':
            fov_voxels = fov_grid_coords[
                (fov_grid_coords[:, 3] >= 0) & (fov_grid_coords[:, 3] < 17)
            ]
        elif dataset == 'kitti360':
            fov_voxels = fov_grid_coords[
                (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 19)
            ]
        else:
            fov_voxels = fov_grid_coords[
                (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
            ]
    print(len(fov_voxels))
    
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # Draw occupied inside FOV voxels
    voxel_size = sum(voxel_size) / 3
    if not sem:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            -fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            colormap="jet",
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
        )
    else:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            -fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=vmin,
            vmax=vmax, # 16
        )

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    if sem:
        if dataset == 'nusc':
            colors = np.array(
                [
                    [  0,   0,   0, 255],       # others
                    [255, 120,  50, 255],       # barrier              orange
                    [255, 192, 203, 255],       # bicycle              pink
                    [255, 255,   0, 255],       # bus                  yellow
                    [  0, 150, 245, 255],       # car                  blue
                    [  0, 255, 255, 255],       # construction_vehicle cyan
                    [255, 127,   0, 255],       # motorcycle           dark orange
                    [255,   0,   0, 255],       # pedestrian           red
                    [255, 240, 150, 255],       # traffic_cone         light yellow
                    [135,  60,   0, 255],       # trailer              brown
                    [160,  32, 240, 255],       # truck                purple                
                    [255,   0, 255, 255],       # driveable_surface    dark pink
                    # [175,   0,  75, 255],       # other_flat           dark red
                    [139, 137, 137, 255],
                    [ 75,   0,  75, 255],       # sidewalk             dard purple
                    [150, 240,  80, 255],       # terrain              light green          
                    [230, 230, 250, 255],       # manmade              white
                    [  0, 175,   0, 255],       # vegetation           green
                    # [  0, 255, 127, 255],       # ego car              dark cyan
                    # [255,  99,  71, 255],       # ego car
                    # [  0, 191, 255, 255]        # ego car
                ]
            ).astype(np.uint8)
        elif dataset == 'kitti360':
            colors = (get_kitti360_colormap()[1:, :] * 255).astype(np.uint8)
        else:
            colors = (get_kitti_colormap()[1:, :] * 255).astype(np.uint8)

        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    
    scene = figure.scene
    scene.camera.position = [118.7195754824976, 118.70290907014409, 120.11124225247899]
    scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [114.42016931210819, 320.9039783052695]
    scene.camera.compute_view_plane_normal()
    scene.render()
    scene.camera.azimuth(-5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(-5)
    scene.render()
    scene.camera.position = [-138.7379881436844, -0.008333206176756428, 99.5084646673331]
    scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [104.37185230017721, 252.84608651497263]
    scene.camera.compute_view_plane_normal()
    scene.render()
    scene.camera.position = [-114.65804807470022, -0.008333206176756668, 82.48137575398867]
    scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [75.17498702830105, 222.91192666552377]
    scene.camera.compute_view_plane_normal()
    scene.render()
    scene.camera.position = [-94.75727115818437, -0.008333206176756867, 68.40940144543957]
    scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [51.04534630774225, 198.1729515833347]
    scene.camera.compute_view_plane_normal()
    scene.render()
    scene.camera.elevation(5)
    scene.camera.orthogonalize_view_up()
    scene.render()
    scene.camera.position = [-107.15500034628069, -0.008333206176756742, 92.16667026873841]
    scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.6463156430702276, -6.454925414290924e-18, 0.7630701733934554]
    scene.camera.clipping_range = [78.84362692774403, 218.2948716014858]
    scene.camera.compute_view_plane_normal()
    scene.render()
    scene.camera.position = [-107.15500034628069, -0.008333206176756742, 92.16667026873841]
    scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.6463156430702277, -6.4549254142909245e-18, 0.7630701733934555]
    scene.camera.clipping_range = [78.84362692774403, 218.2948716014858]
    scene.camera.compute_view_plane_normal()
    scene.render()
    scene.camera.elevation(5)
    scene.camera.orthogonalize_view_up()
    scene.render()
    scene.camera.elevation(5)
    scene.camera.orthogonalize_view_up()
    scene.render()
    scene.camera.elevation(-5)
    mlab.pitch(-8)
    mlab.move(up=15)
    scene.camera.orthogonalize_view_up()
    scene.render()

    # scene.camera.position = [  0.75131739, -35.08337438,  16.71378558]
    # scene.camera.focal_point = [  0.75131739, -34.21734897,  16.21378558]
    # scene.camera.view_angle = 40.0
    # scene.camera.view_up = [0.0, 0.0, 1.0]
    # scene.camera.clipping_range = [0.01, 300.]
    # scene.camera.compute_view_plane_normal()
    # scene.render()

    filepath = os.path.join(save_dir, f'{name}.png')
    if offscreen:
        mlab.savefig(filepath)
    else:
        mlab.show()
    mlab.close()

def get_nuscenes_colormap():
    colors = np.array(
        [
            [  0,   0,   0, 255],       # others
            [255, 120,  50, 255],       # barrier              orange
            [255, 192, 203, 255],       # bicycle              pink
            [255, 255,   0, 255],       # bus                  yellow
            [  0, 150, 245, 255],       # car                  blue
            [  0, 255, 255, 255],       # construction_vehicle cyan
            [255, 127,   0, 255],       # motorcycle           dark orange
            [255,   0,   0, 255],       # pedestrian           red
            [255, 240, 150, 255],       # traffic_cone         light yellow
            [135,  60,   0, 255],       # trailer              brown
            [160,  32, 240, 255],       # truck                purple                
            [255,   0, 255, 255],       # driveable_surface    dark pink
            # [175,   0,  75, 255],       # other_flat           dark red
            [139, 137, 137, 255],
            [ 75,   0,  75, 255],       # sidewalk             dard purple
            [150, 240,  80, 255],       # terrain              light green          
            [230, 230, 250, 255],       # manmade              white
            [  0, 175,   0, 255],       # vegetation           green
            # [  0, 255, 127, 255],       # ego car              dark cyan
            # [255,  99,  71, 255],       # ego car
            # [  0, 191, 255, 255]        # ego car
        ]
    ).astype(np.float32) / 255.
    return colors

def save_gaussian(save_dir, gaussian, name):
    empty_label = 17
    sem_cmap = get_nuscenes_colormap()

    torch.save(gaussian, os.path.join(save_dir, f'{name}_attr.pth'))

    means = gaussian.means[0].detach().cpu().numpy() # g, 3
    scales = gaussian.scales[0].detach().cpu().numpy() # g, 3
    rotations = gaussian.rotations[0].detach().cpu().numpy() # g, 4
    opas = gaussian.opacities[0]
    if opas.numel() == 0:
        opas = torch.ones_like(gaussian.means[0][..., :1])
    opas = opas.squeeze().detach().cpu().numpy() # g
    sems = gaussian.semantics[0].detach().cpu().numpy() # g, 18
    pred = np.argmax(sems, axis=-1)

    mask = (pred != empty_label) & (opas > 0.75)

    means = means[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    opas = opas[mask]
    pred = pred[mask]

    # number of ellipsoids 
    ellipNumber = means.shape[0]

    #set colour map so each ellipsoid as a unique colour
    norm = colors.Normalize(vmin=-1.0, vmax=5.4)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.figure(figsize=(9, 9), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=46, azim=-180)
    scalar = 2

    # compute each and plot each ellipsoid iteratively
    border = np.array([
        [-50.0, -50.0, 0.0],
        [-50.0, 50.0, 0.0],
        [50.0, -50.0, 0.0],
        [50.0, 50.0, 0.0],
    ])
    ax.plot_surface(border[:, 0:1], border[:, 1:2], border[:, 2:], 
        rstride=1, cstride=1, color=[0, 0, 0, 1], linewidth=0, alpha=0., shade=True)

    for indx in range(ellipNumber):
        
        center = means[indx]
        radii = scales[indx] * scalar
        rot_matrix = rotations[indx]
        rot_matrix = Quaternion(rot_matrix).rotation_matrix.T

        # calculate cartesian coordinates for the ellipsoid surface
        u = np.linspace(0.0, 2.0 * np.pi, 10)
        v = np.linspace(0.0, np.pi, 10)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        xyz = np.stack([x, y, z], axis=-1) # phi, theta, 3
        xyz = rot_matrix[None, None, ...] @ xyz[..., None]
        xyz = np.squeeze(xyz, axis=-1)

        xyz = xyz + center[None, None, ...]

        ax.plot_surface(
            xyz[..., 1], -xyz[..., 0], xyz[..., 2], 
            rstride=1, cstride=1, color=sem_cmap[pred[indx]], linewidth=0, alpha=opas[indx], shade=True)

    plt.axis("equal")
    # plt.gca().set_box_aspect([1, 1, 1])
    ax.grid(False)
    ax.set_axis_off()    

    filepath = os.path.join(save_dir, f'{name}.png')
    plt.savefig(filepath)

    plt.cla()
    plt.clf()

def denormalize_3d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    new_pts[...,2:3] = (pts[...,2:3]*(pc_range[5] -
                            pc_range[2]) + pc_range[2])
    return new_pts

def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])

    return bboxes

def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    return new_pts

class MapTRNMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 z_cfg = dict(
                    pred_z_flag=False,
                    gt_z_flag=False,
                 ),
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

        self.z_cfg = z_cfg

    def encode(self):

        pass

    def decode_single(self, cls_scores, bbox_preds, pts_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
            pts_preds (Tensor):
                Shape [num_query, fixed_num_pts, 2]
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]
        pts_preds = pts_preds[bbox_index]
       
        final_box_preds = denormalize_2d_bbox(bbox_preds, self.pc_range) 
        #num_q,num_p,2
        final_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range) if not self.z_cfg['gt_z_flag'] \
                        else denormalize_3d_pts(pts_preds, self.pc_range) 
        # final_box_preds = bbox_preds 
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :4] >=
                    self.post_center_range[:4]).all(1)
            mask &= (final_box_preds[..., :4] <=
                     self.post_center_range[4:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            pts = final_pts_preds[mask]
            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
                'pts': pts,
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        all_pts_preds = preds_dicts['all_pts_preds'][-1]
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i],all_pts_preds[i]))
        return predictions_list

save_map_index = 0

def vis_map_train(pred, gt, save_dir = "test_map"):
    global save_map_index

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    def draw(instances, labels, filename="map.png", is_gt=False):
        plt.figure(figsize=(2, 4))
        pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')
        colors_plt = ['orange', 'b', 'r'] # ['divider', 'ped_crossing', 'boundary']
        if is_gt:
            instances = instances[0].fixed_num_sampled_points
        for points, label in zip(instances, labels):
            pts = points.numpy()
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            plt.plot(x, y, color=colors_plt[label],linewidth=1,alpha=0.8,zorder=-1)
            plt.scatter(x, y, color=colors_plt[label],s=2,alpha=0.8,zorder=-1)
        plt.savefig(f'{save_dir}/{filename}', bbox_inches='tight', format='png',dpi=1200)
        plt.close()

    draw(gt["gt_bboxes_3d"], gt["gt_labels_3d"][0], filename=f"map_gt_{save_map_index}.png", is_gt=True)

    def decode(preds_dicts):
        bbox_coder_args=dict(
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
            max_num=50,
            voxel_size=[0.5, 0.5, 0.5],
            num_classes=3)
        bbox_coder = MapTRNMSFreeCoder(**bbox_coder_args)
        preds_dicts = bbox_coder.decode(preds_dicts)

        preds = preds_dicts[0]
        keep = preds['scores'] > 0.3
        pts = preds['pts'][keep].to('cpu').detach()
        labels = preds['labels'][keep].cpu().detach()

        return pts, labels

    draw(*decode(pred), filename=f"map_pred_{save_map_index}.png")

    save_map_index += 1