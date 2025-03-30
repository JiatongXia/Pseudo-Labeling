import os
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import math
import random
import cv2

from run_nerf_helpers import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None, fars=None, nears=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    if nears is not None:
        near = nears.unsqueeze(-1)
        far = fars.unsqueeze(-1)
    else:
        near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, rgbsavedir=None, depthsavedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
            
    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, depth, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        if rgbsavedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(rgbsavedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
        
        if depthsavedir is not None:
            depth = depth.cpu().numpy()
            x = np.nan_to_num(depth)
            mi = np.min(x)
            ma = np.max(x)
            x[x>ma] = ma
            x = (x-mi)/(ma-mi+1e-8)  # normalize to 0~1
            x = (255*x).astype(np.uint8)
            depth_rgb = cv2.applyColorMap(x, cv2.COLORMAP_JET)

            filename = os.path.join(depthsavedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, depth_rgb)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())
    
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']

        # ############################
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2) 

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] 
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] 

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False) 

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')
    
    #######################
    parser.add_argument("--N_iters", type=int, default=200000, 
                        help='number of iters')
    parser.add_argument("--ft_iters", type=int, default=10000, 
                        help='number of iters')
    parser.add_argument("--use_R_change", action='store_true')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    images, poses, H, W, focal, near, far, i_split = load_scene(args.datadir)   # i_split: train, test
    i_train, i_test, i_video = i_split

    train_poses = poses[i_train,:3,:]
    test_poses = poses[i_test,:3,:]
    video_poses = poses[i_video,:3,:]

    len_train_and_test = len(i_train) + len(i_test)
    poses = poses[:len_train_and_test]
    
    hwf = [H, W, focal]

    print('NEAR FAR', near, far)

    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

    use_R_change = args.use_R_change

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)


    if args.render_only:
        print('Render Only!')

        print('Render Test Views.')
        rgbsavedir = os.path.join(basedir, expname, 'rgb_test')
        os.makedirs(rgbsavedir, exist_ok=True)
        depthsavedir = os.path.join(basedir, expname, 'depth_test')
        os.makedirs(depthsavedir, exist_ok=True)
        print('test poses shape', poses[i_test].shape)
        with torch.no_grad():
            render_path(torch.Tensor(test_poses).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], rgbsavedir=rgbsavedir, depthsavedir=depthsavedir)
        print('Saved test set')
        

        print('Render Video.')
        videosavedir = os.path.join(basedir, expname, 'video_rgb')
        os.makedirs(videosavedir, exist_ok=True)
        with torch.no_grad():
            rgbs, disps = render_path(torch.Tensor(video_poses).to(device), hwf, K, args.chunk, render_kwargs_test,
                                                    rgbsavedir=videosavedir)
        print('Done, saving', rgbs.shape)
        moviebase = os.path.join(basedir, expname, '{}_spiral'.format(expname))
        imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
        return


    #######################################

    # For random ray batching
    print('get rays')
    rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0)
    print('done, concats')
    rays_rgb = np.concatenate([rays, images[:,None]], 1) 
    rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) 
    rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) 
    rays_rgb = np.reshape(rays_rgb, [-1,3,3]) 
    rays_rgb = rays_rgb.astype(np.float32)
    print('shuffle rays')
    np.random.shuffle(rays_rgb)

    print('done')
    i_batch = 0

    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)

    rays_rgb = torch.Tensor(rays_rgb).to(device)


    ####################################################
    ##################### Training #####################
    ####################################################

    print('Start Training Stages')

    N_rand = args.N_rand
    N_iters = args.N_iters
    ft_iters = args.ft_iters

    print('TRAIN views are', i_train)
    print('TEST views are', i_test)


    start = start + 1
    print('Training Stage 1: Initial training for NeRF.')
    for i in trange(start, N_iters+1):
        # Random over all images
        batch = rays_rgb[i_batch:i_batch+N_rand]
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_s = batch[:2], batch[2]

        i_batch += N_rand
        if i_batch >= rays_rgb.shape[0]:
            print("Shuffle data after an epoch!")
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]
            i_batch = 0

        #####  Core optimization loop  #####
        rgb, disp, acc, depth, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0

        loss.backward()
        optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        global_step += 1
        
        # logging
        if i%N_iters==0 and i!=start:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%N_iters==0 and i!=start:
            rgbsavedir = os.path.join(basedir, expname, 'rgb_{:06d}'.format(i))
            os.makedirs(rgbsavedir, exist_ok=True)
            depthsavedir = os.path.join(basedir, expname, 'depth_{:06d}'.format(i))
            os.makedirs(depthsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], rgbsavedir=rgbsavedir, depthsavedir=depthsavedir)
            print('Saved test set')

        if i%args.i_print==0 and i!=start:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        
    # Train depths
    train_depth_path = os.path.join(args.datadir, 'train_depth.npz')
    train_rgb_dir = os.path.join(basedir, expname, 'train_rgbs')
    train_depth_dir = os.path.join(basedir, expname, 'train_depths')
    os.makedirs(train_rgb_dir, exist_ok=True)
    os.makedirs(train_depth_dir, exist_ok=True)
    if os.path.exists(train_depth_path):
        print('Found Train Depths.')
        train_depth_npz = np.load(train_depth_path)
        train_depths = train_depth_npz['depths']
    else:
        print('Rendering Train Depths.')
        with torch.no_grad():
            train_depths = render_depth_train(torch.Tensor(poses[i_train]).to(device), hwf, K, args.chunk, render_kwargs_test, img_savedir=train_rgb_dir, depth_savedir=train_depth_dir, npz_savedir=train_depth_path, render_factor=2)

    train_depths = torch.Tensor(train_depths).to(device)

    print('Training Stage 2: Fine-tuning for close-up region.')
    start = global_step + 1
    for i in trange(start, (N_iters+ft_iters+1)):

        # Random over all images
        N_far = int(N_rand * 1.5)
        N_near = int(N_rand * 0.5 * 1.1)

        batch = rays_rgb[i_batch:i_batch+N_far] 
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_s = batch[:2], batch[2]
        fars = torch.ones(target_s.shape[0])
        fars = fars * far
        nears = torch.ones(target_s.shape[0])
        nears = nears * near

        i_batch += N_far
        if i_batch >= rays_rgb.shape[0]:
            print("Shuffle data after an epoch!")
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]
            i_batch = 0


        ####################### generate virtual pose  ########################
    
        i_near = range(len(i_train))
        virtual_i = np.random.choice(i_near)
    
        pose = poses[i_train[virtual_i]]
        depth = train_depths[virtual_i]
        virtual_pose = generate_virtual_pose(pose, depth, hwf, K, use_R_change)


        ########### select and render random rays from virtual pose  ###########
        
        rays_o, rays_d = get_rays_torch(H, W, K, virtual_pose[:3,:4])

        select_mask = torch.ones(H, W).to(device)   # selcet random indices
        mask_coord = torch.nonzero(select_mask)
        select_idx = np.random.randint(0, (H*W), N_near)
        select_coord = mask_coord[select_idx]

        rays_o = rays_o[select_coord[:,0], select_coord[:,1]]
        rays_d = rays_d[select_coord[:,0], select_coord[:,1]]

        virtual_batch_rays = torch.stack([rays_o, rays_d], 0)
        virtual_fars = torch.ones(rays_o.shape[0])
        virtual_fars = virtual_fars * far
        virtual_nears = torch.ones(rays_o.shape[0])
        virtual_nears = virtual_nears * near

        with torch.no_grad():
            rgb_, disp_, acc_, depth_virtual, extras_ = render(H, W, K, chunk=args.chunk, rays=virtual_batch_rays,
                                    verbose=i < 10, retraw=True, fars=virtual_fars, nears=virtual_nears,
                                    **render_kwargs_train)

        scoord_3dw = rays_o + rays_d * depth_virtual[...,None]

        i_img = i_train[virtual_i]
        pose1 = poses[i_img]
        image1 = images[i_img]
        depth1 = train_depths[virtual_i]

        warped_rgb, warped_mask = warp_from_virtual(pose1, image1, depth1, scoord_3dw, hwf, select_coord)


        ################################# warp from training images

        projected_rgb, projected_mask = warp_train(virtual_pose, poses, images, train_depths, hwf, K, i_train, select_coord)


        warped_rgb = warped_rgb[select_coord[:,0], select_coord[:,1]]
        projected_rgb = projected_rgb[select_coord[:,0], select_coord[:,1]]

        diff_rgb = projected_rgb - warped_rgb

        d_rgb = 0.1
        r_mask = (-d_rgb < diff_rgb[...,0]) * (diff_rgb[...,0] < d_rgb)
        g_mask = (-d_rgb < diff_rgb[...,1]) * (diff_rgb[...,1] < d_rgb)
        b_mask = (-d_rgb < diff_rgb[...,2]) * (diff_rgb[...,2] < d_rgb) 
        rgb_mask = r_mask * g_mask * b_mask

        mask = warped_mask * projected_mask * rgb_mask

        virtual_target = projected_rgb[mask]

        virtual_batch_rays = virtual_batch_rays.permute(1,0,2)[mask].permute(1,0,2)
        virtual_fars = virtual_fars[mask]
        virtual_nears = virtual_nears[mask]

        #################################################################

        batch_rays = torch.cat([batch_rays, virtual_batch_rays], 1)
        target_s = torch.cat([target_s, virtual_target], 0)
        fars = torch.cat([fars, virtual_fars], 0)
        nears = torch.cat([nears, virtual_nears], 0)


        #####  Core optimization loop  #####
        rgb, disp, acc, depth, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True, fars=fars, nears=nears,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0

        loss.backward()
        optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################
            
        global_step += 1

        # logging
        if i%(N_iters+ft_iters)==0 and i!=start:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)
        
        if i%(N_iters+ft_iters)==0 and i!=start:
            rgbsavedir = os.path.join(basedir, expname, 'rgb_{:06d}'.format(i))
            os.makedirs(rgbsavedir, exist_ok=True)
            depthsavedir = os.path.join(basedir, expname, 'depth_{:06d}'.format(i))
            os.makedirs(depthsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(test_poses).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], rgbsavedir=rgbsavedir, depthsavedir=depthsavedir)
            print('Saved test set')
        
        if i%args.i_print==0 and i!=start:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        if i%args.i_video==0 and i!=start:
            videosavedir = os.path.join(basedir, expname, 'video_rgb_{:06d}'.format(i))
            os.makedirs(videosavedir, exist_ok=True)
            with torch.no_grad():
                rgbs, disps = render_path(torch.Tensor(video_poses).to(device), hwf, K, args.chunk, render_kwargs_test,
                                                        rgbsavedir=videosavedir)
            print('Done, saving', rgbs.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)


def render_depth_train(render_poses, hwf, K, chunk, render_kwargs, img_savedir=None, depth_savedir=None, npz_savedir=None, render_factor=0):

    H, W, focal = hwf
    
    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    depths = []
    for i, c2w in enumerate(tqdm(render_poses)):
        rgb, disp, acc, depth, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgb = rgb.cpu().numpy()
        depth = depth.cpu().numpy()
        depth = cv2.resize(depth, (W*2, H*2))
        depths.append(depth)

        x = np.nan_to_num(depth)
        mi = np.min(x)
        ma = np.max(x)
        x[x>ma] = ma
        x = (x-mi)/(ma-mi+1e-8)  # normalize to 0~1
        x = (255*x).astype(np.uint8)
        depth_rgb = cv2.applyColorMap(x, cv2.COLORMAP_JET)

        if img_savedir is not None:
            rgb8 = to8b(rgb)
            img_filename = os.path.join(img_savedir, '{:03d}.png'.format(i))
            imageio.imwrite(img_filename, rgb8)

        if depth_savedir is not None:
            depth_filename = os.path.join(depth_savedir, '{:03d}.png'.format(i))
            imageio.imwrite(depth_filename, depth_rgb)

    depths = np.stack(depths, 0)
    if npz_savedir is not None:
        np.savez(npz_savedir, depths=depths)
    return depths


def get_rays_torch(H, W, K, c2w):
    K = torch.Tensor(K).to(device)
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing='xy')
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1]
    rays_o = rays_o.repeat(H,W,1)
    return rays_o, rays_d


def generate_virtual_pose(pose, depth, hwf, K, use_R_change):

    H, W, focal = hwf

    # ensure the anchor point not too peripheral
    depth_mean = depth.mean()
    mean_mask = torch.logical_and(((0.8*depth_mean) < depth), (depth < (1.2*depth_mean)))

    select_coord = torch.nonzero(mean_mask)
    random_coord = np.random.choice(range(select_coord.shape[0]))

    v_row = select_coord[random_coord][0]
    v_col = select_coord[random_coord][1]

    rays_o, rays_d = get_rays_torch(H, W, K, pose[:3,:4])
    scoord_3dw = rays_o + rays_d * depth[...,None]

    t0 = rays_o[0][0]   # far
    t1 = scoord_3dw[v_row, v_col]  # target

    R0 = pose[:,:3]    # far

    x0, y0, z0 = rotationMatrixToEulerAngles(R0) 

    t = (3*t1 + t0) / 4

    dx = random.uniform(-1,1)*math.pi/4
    dy = random.uniform(-1,1)*math.pi/4
    dz = random.uniform(-1,1)*math.pi/4

    # x
    if abs(x0+dx) < math.pi:   
        x = x0 + dx
    else:
        if x0+dx < -math.pi:
            d = x0 + dx + math.pi  # over pi
            x = math.pi + d
        else:    
            d = x0 + dx - math.pi  # over pi
            x = -math.pi + d
    # y
    if abs(y0+dy) < math.pi:   
        y = y0 + dy
    else:
        if y0+dy < -math.pi:
            d = y0 + dy + math.pi  # over pi
            y = math.pi + d
        else:    
            d = y0 + dy - math.pi  # over pi
            y = -math.pi + d
    # z
    if abs(z0+dz) < math.pi:   
        z = z0 + dz
    else:
        if z0+dz < -math.pi:
            d = z0 + dz + math.pi  # over pi
            z = math.pi + d
        else:    
            d = z0 + dz - math.pi  # over pi
            z = -math.pi + d
    
    if use_R_change:
        e = [x,y,z]
        R = torch.Tensor(eulerAnglesToRotationMatrix(e)).to(device)
    else:
        R = R0
        
    virtual_pose = torch.cat([R,t.unsqueeze(-1)],1)
    return virtual_pose


def warp_from_virtual(pose, image, depth, scoord_3dw, hwf, select_coord):
    
    H, W, focal = hwf

    t = pose[:3,3]
    R = pose[:3,:3]

    t = t.repeat(scoord_3dw.shape[0],1)
    RI = torch.transpose(R, -1,-2)

    scoord_3dts = scoord_3dw - t

    scoord_3dts = RI[None,:,:] @ scoord_3dts[...,None]
    scoord_3dts = scoord_3dts.squeeze(-1)

    z_ts = -scoord_3dts[..., 2]
    y_ts = -scoord_3dts[..., 1]
    x_ts = scoord_3dts[..., 0]

    col = (focal * x_ts) / z_ts + (0.5 * W)
    row = (focal * y_ts) / z_ts + (0.5 * H)


    col_int = torch.round(col)
    row_int = torch.round(row)
    warp_coords = torch.cat((col_int.unsqueeze(-1), row_int.unsqueeze(-1)), -1)

    select_warp_coords = torch.zeros((H,W,2))
    select_warp_coords[select_coord[:,0], select_coord[:,1]] = warp_coords
                
    pix_coords = select_warp_coords
    pix_coords[..., 0] /= W - 1
    pix_coords[..., 1] /= H - 1
    pix_coords = (pix_coords - 0.5) * 2

    depth = depth.unsqueeze(0).unsqueeze(0)
    image = image.unsqueeze(0).permute(0,3,1,2)

    pix_coords = pix_coords.unsqueeze(0)

    warped_img = F.grid_sample(image, pix_coords.float(), padding_mode='zeros', align_corners=False)
    warped_img = warped_img.squeeze(0).permute(1,2,0)

    warped_depth = F.grid_sample(depth, pix_coords.float(), padding_mode='zeros', align_corners=False)
    warped_depth = warped_depth.squeeze(0).squeeze(0)   # [H,W]
    warped_depth = warped_depth[select_coord[:,0], select_coord[:,1]]

    null_mask = warped_depth==0
    occ_mask = warped_depth < 0.95*z_ts    # occlusion region
    pen_mask = warped_depth > 1.05*z_ts    # penetrate region
    warped_mask = torch.logical_or(torch.logical_or(occ_mask, pen_mask), null_mask)
    warped_mask = torch.logical_not(warped_mask)      # non-occlusion

    return warped_img, warped_mask


def warp_train(virtual_pose, poses, images, train_depths, hwf, K, i_train, select_coord):

    H, W, focal = hwf

    # NOTE 
    c0 = select_coord
    select_coord_idx = (c0[:,0] + c0[:,1])*(c0[:,0] + c0[:,1] +1) / 2 + c0[:,1]
    
    t = virtual_pose[:3,3]
    R = virtual_pose[:3,:3]

    t = t.repeat(H,W,1)
    RI = torch.transpose(R, -1,-2)

    projected_rgb = torch.zeros((H,W,3)).to(device)
    projected_depth = torch.zeros((H,W)).to(device)

    for jj in range(len(i_train)):

        pose = poses[i_train[jj]]
        image = images[i_train[jj]]
        depth = train_depths[jj]

        rays_o, rays_d = get_rays_torch(H, W, K, pose[:3,:4])
        scoord_3dw = rays_o + rays_d * depth[...,None]

        scoord_3dts = scoord_3dw - t
        scoord_3dts = RI[None,None,:,:] @ scoord_3dts[...,None]
        scoord_3dts = scoord_3dts.squeeze(-1)

        z_ts = -scoord_3dts[..., 2]
        y_ts = -scoord_3dts[..., 1]
        x_ts = scoord_3dts[..., 0]

        col = (focal * x_ts) / z_ts + (0.5 * W)
        row = (focal * y_ts) / z_ts + (0.5 * H)

        col_int = torch.round(col).long()
        row_int = torch.round(row).long()
        coords = torch.cat((row_int.unsqueeze(-1), col_int.unsqueeze(-1)), -1)
        mask_coords = (col_int > 0) * (row_int > 0) * (col_int < W - 1) * (row_int < H - 1)

        original_img_coord = torch.nonzero(mask_coords)
        projected_2d_coord = coords[mask_coords]
        
        u1 = original_img_coord[:, 0]
        v1 = original_img_coord[:, 1]

        u0 = projected_2d_coord[:, 0]
        v0 = projected_2d_coord[:, 1]

        # NOTE
        projected_idx = (u0 + v0)*(u0 + v0 +1) / 2 + v0
        mask_coord = torch.isin(projected_idx, select_coord_idx)

        mask_null = projected_depth[u0, v0] == 0
        mask_occlusion = 0.95*projected_depth[u0, v0] > z_ts[u1, v1]
        mask_warp = torch.logical_or(mask_null, mask_occlusion)
        mask_warp = mask_warp * mask_coord

        u1 = u1[mask_warp]
        v1 = v1[mask_warp]
        u0 = u0[mask_warp]
        v0 = v0[mask_warp]

        projected_depth[u0, v0] = z_ts[u1, v1]
        projected_rgb[u0, v0] = image[u1, v1]   # projected result

    projected_mask = projected_depth != 0
    projected_mask = projected_mask[select_coord[:,0], select_coord[:,1]]

    return projected_rgb, projected_mask


def load_scene(basedir):
    splits = ['train', 'test', 'video']

    all_imgs = []
    all_poses = []
    counts = [0]
    filenames = []
    for s in splits:
        if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format(s))):
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                meta = json.load(fp)

            near = float(meta['near'])
            far = float(meta['far'])
           
            imgs = []
            poses = []
            
            for frame in meta['frames']:
                if len(frame['file_path']) != 0:
                    img = imageio.imread(os.path.join(basedir, frame['file_path'])) / 255.
                    imgs.append(img)
                    filenames.append(frame['file_path'])
                    
                poses.append(np.array(frame['transform_matrix']))
                fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
                H, W = int(cy*2), int(cx*2)
                
            counts.append(counts[-1] + len(poses))

            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
            all_poses.append(np.array(poses).astype(np.float32))

        else:
            counts.append(counts[-1])
        
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    return imgs, poses, H, W, fx, near, far, i_split


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
