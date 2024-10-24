# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp

def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)


def get_fps(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            a, b = line.decode().strip().split('/')
            return int(a) / int(b)


def read_video(filename, skip=0, limit=-1):
    w, h = get_resolution(filename)
    # w = 1000
    # h = 1002

    command = ['ffmpeg',
               '-i', filename,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vsync', '0',
               '-vcodec', 'rawvideo', '-']

    i = 0
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        while True:
            data = pipe.stdout.read(w * h * 3)
            if not data:
                break
            i += 1
            if i > limit and limit != -1:
                continue
            if i > skip:
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))


def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


def render_animation(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0, newpose=None):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    plt.ioff()
    if newpose is not None:
        fig = plt.figure(figsize=(size * (1 + len(poses) + len(newpose)), size))
        ax_in = fig.add_subplot(1, 1 + len(poses) + len(newpose), 1)
    else:
        fig = plt.figure(figsize=(size * (1 + len(poses)), size))
        ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    if newpose is not None:
        axnew = fig.add_subplot(1, 1 + len(poses) + len(newpose), 2, projection='3d')
        axnew.view_init(elev=15., azim=azim)
        axnew.set_xlim3d([-radius / 2, radius / 2])
        axnew.set_zlim3d([0, radius])
        axnew.set_ylim3d([-radius / 2, radius / 2])
        try:
            axnew.set_aspect('equal')
        except NotImplementedError:
            axnew.set_aspect('auto')
        axnew.set_xticklabels([])
        axnew.set_yticklabels([])
        axnew.set_zticklabels([])
        axnew.dist = 7.5
        axnew.set_title('PoseFormer') #, pad=35
        ax_3d.append(axnew)
        lines_3d.append([])
        trajectories.append(newpose[:, 0, [0, 1]])

    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title) #, pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())

    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

        keypoints = keypoints[input_video_skip:] # todo remove
        for idx in range(len(poses)):
            poses[idx] = poses[idx][input_video_skip:]
        if newpose is not None:
            newpose = newpose[input_video_skip:]

        if fps is None:
            fps = get_fps(input_video_path)

    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        if newpose is not None:
            newpose = downsample_tensor(newpose, downsample)
            for idx in range(len(poses)+len(newpose)):
                poses[idx] = downsample_tensor(poses[idx], downsample)
                trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        else:
            for idx in range(len(poses)):
                poses[idx] = downsample_tensor(poses[idx], downsample)
                trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None

    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    parents = skeleton.parents()
    def update_video(i):
        nonlocal initialized, image, lines, points

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])

        # Update 2D poses
        joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
        colors_2d = np.full(keypoints.shape[1], 'black')
        colors_2d[joints_right_2d] = 'red'
        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect='equal')

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1]:
                    # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                    lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))

                col = 'orange' if j in skeleton.joints_right() else 'green'
                
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
            # Plot 2D keypoints
            points = ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)

            initialized = True
        else:
            image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1]:
                    lines[j-1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                             [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

                # Plot 2D keypoints
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j-1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                    lines_3d[n][j-1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                    lines_3d[n][j-1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')
            # Plot 2D keypoints
            points.set_offsets(keypoints[i])

        print('{}/{}      '.format(i, limit), end='\r')


    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000/fps, repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()




######################################

def render_animation_overlap(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0, with_club=False):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    titles = ["Reconstruction", "Ground Truth", "Merge"]

    plt.ioff()
    fig = plt.figure(figsize=(size*(1 + len(poses)), size))
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    # ax_in.set_title('Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 2

    def get_ax(title, position):
        ax = fig.add_subplot(1, 1 + len(poses), position, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([0, radius])
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title) #, pad=35

        # ax.set_xlabel("x axis")
        # ax.set_ylabel("y axis")

        return ax

    for i, title in enumerate(titles):
        ax_3d.append(get_ax(title, i+1))
        lines_3d.append([])

        if i < 2:
            items = poses.items()
            _, data = list(items)[i]
            print(_)

            temp = np.copy(data[:, :, 1])
            data[:, :, 1] = data[:, :, 0]
            data[:, :, 0] = temp
            trajectories.append(data[:, 0, [0, 1]])

            
    poses = list(poses.values())
    

    initialized = False
    if not with_club:
        parents = skeleton.parents()[:17]
    else:
        parents = skeleton.parents()

    limit = len(poses[0])
    

    def update_video(i):
        nonlocal initialized

        for idx, ax in enumerate(ax_3d):
            text = f"{titles[idx]}: {i}/{limit}"
            if idx == 2:
                text += "  dash(GT)"
            ax.set_title(text)


        for n, ax in enumerate(ax_3d):
            a = n if n < 2 else n-1
            
            ax.set_xlim3d([-radius/2 + trajectories[a][i, 0], radius/2 + trajectories[a][i, 0]])
            ax.set_ylim3d([-radius/2 + trajectories[a][i, 1], radius/2 + trajectories[a][i, 1]])

        if not initialized:
            
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                
                if j in skeleton.joints_right():
                    col = '#FF2D2D'
                elif j in [17, 18, 19, 20, 21]:
                    col =  '#46A3FF'
                else:
                    col = '#6C6C6C'
                
                for n, ax in enumerate(ax_3d):
                    if n < 2:
                        pos = poses[n][i]
                        lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                                [pos[j, 1], pos[j_parent, 1]],
                                                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
                    else:
                        pos_pred = poses[0][i]
                        pos_gt = poses[1][i]

                        lines_3d[n].append([
                            ax.plot([pos_pred[j, 0], pos_pred[j_parent, 0]], [pos_pred[j, 1], pos_pred[j_parent, 1]], [pos_pred[j, 2], pos_pred[j_parent, 2]], zdir='z', c=col),
                            ax.plot([pos_gt[j, 0], pos_gt[j_parent, 0]], [pos_gt[j, 1], pos_gt[j_parent, 1]], [pos_gt[j, 2], pos_gt[j_parent, 2]], zdir='z', c=col, linestyle='dashed'),
                        ])
            
            initialized = True
        else:
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
            
                
                for n, ax in enumerate(ax_3d):
                    if n < 2:
                        pos = poses[n][i]
                        lines_3d[n][j-1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                        lines_3d[n][j-1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                        lines_3d[n][j-1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')
                    else:
                        pos_pred = poses[0][i]
                        pos_gt = poses[1][i]

                        lines_3d[n][j-1][0][0].set_xdata(np.array([pos_pred[j, 0], pos_pred[j_parent, 0]]))
                        lines_3d[n][j-1][0][0].set_ydata(np.array([pos_pred[j, 1], pos_pred[j_parent, 1]]))
                        lines_3d[n][j-1][0][0].set_3d_properties(np.array([pos_pred[j, 2], pos_pred[j_parent, 2]]), zdir='z')

                        lines_3d[n][j-1][1][0].set_xdata(np.array([pos_gt[j, 0], pos_gt[j_parent, 0]]))
                        lines_3d[n][j-1][1][0].set_ydata(np.array([pos_gt[j, 1], pos_gt[j_parent, 1]]))
                        lines_3d[n][j-1][1][0].set_3d_properties(np.array([pos_gt[j, 2], pos_gt[j_parent, 2]]), zdir='z')

        
        print('{}/{}      '.format(i, limit), end='\r')
        

    fig.tight_layout()
    
    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000/fps, repeat=False)
    if with_club:
        output = f"club_{output}"
    else:
        output = f"no_club_{output}"

    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=300, writer='imagemagick')
    # elif output.endswith('.pdf'):
    #     for i in range(limit):
    #         # anim.save(f'pdf_res/{i}_ani.pdf', dpi=300, fps=30, writer='matplotlib.animation.PdfWriter')
    #         anim.save(f'pdf_res/{i}_ani.png', dpi=300, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()




def render_animation_inference(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0, with_club=False):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    titles = ["Reconstruction"]

    plt.ioff()
    fig = plt.figure(figsize=(size*(1 + len(poses)), size))
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    # ax_in.set_title('Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 2

    def get_ax(title, position):
        ax = fig.add_subplot(1, 1 + len(poses), position, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([0, radius])
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title) #, pad=35

        # ax.set_xlabel("x axis")
        # ax.set_ylabel("y axis")

        return ax

    for i, title in enumerate(titles):
        ax_3d.append(get_ax(title, i+1))
        lines_3d.append([])

        if i < 2:
            items = poses.items()
            _, data = list(items)[i]
            print(_)

            temp = np.copy(data[:, :, 1])
            data[:, :, 1] = data[:, :, 0]
            data[:, :, 0] = temp
            trajectories.append(data[:, 0, [0, 1]])

            
    poses = list(poses.values())
    

    initialized = False
    if not with_club:
        parents = skeleton.parents()[:17]
    else:
        parents = skeleton.parents()

    limit = len(poses[0])
    

    def update_video(i):
        nonlocal initialized

        for idx, ax in enumerate(ax_3d):
            text = f"{titles[idx]}: {i+1}/{limit}"
            if idx == 2:
                text += "  dash(GT)"
            ax.set_title(text)


        for n, ax in enumerate(ax_3d):
            a = n if n < 2 else n-1
            
            ax.set_xlim3d([-radius/2 + trajectories[a][i, 0], radius/2 + trajectories[a][i, 0]])
            ax.set_ylim3d([-radius/2 + trajectories[a][i, 1], radius/2 + trajectories[a][i, 1]])

        if not initialized:
            
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                
                if j in skeleton.joints_right():
                    col = '#FF2D2D'
                elif j in skeleton.joints_left():
                    col = '#2D6C2D'
                elif j in [17, 18, 19, 20, 21]:
                    col =  '#46A3FF'
                else:
                    col = '#6C6C6C'
                
                for n, ax in enumerate(ax_3d):
                    if n < 2:
                        pos = poses[n][i]
                        lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                                [pos[j, 1], pos[j_parent, 1]],
                                                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
                    else:
                        pos_pred = poses[0][i]
                        pos_gt = poses[1][i]

                        lines_3d[n].append([
                            ax.plot([pos_pred[j, 0], pos_pred[j_parent, 0]], [pos_pred[j, 1], pos_pred[j_parent, 1]], [pos_pred[j, 2], pos_pred[j_parent, 2]], zdir='z', c=col),
                            ax.plot([pos_gt[j, 0], pos_gt[j_parent, 0]], [pos_gt[j, 1], pos_gt[j_parent, 1]], [pos_gt[j, 2], pos_gt[j_parent, 2]], zdir='z', c=col, linestyle='dashed'),
                        ])
            
            initialized = True
        else:
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
            
                
                for n, ax in enumerate(ax_3d):
                    if n < 2:
                        pos = poses[n][i]
                        lines_3d[n][j-1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                        lines_3d[n][j-1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                        lines_3d[n][j-1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')
                    else:
                        pos_pred = poses[0][i]
                        pos_gt = poses[1][i]

                        lines_3d[n][j-1][0][0].set_xdata(np.array([pos_pred[j, 0], pos_pred[j_parent, 0]]))
                        lines_3d[n][j-1][0][0].set_ydata(np.array([pos_pred[j, 1], pos_pred[j_parent, 1]]))
                        lines_3d[n][j-1][0][0].set_3d_properties(np.array([pos_pred[j, 2], pos_pred[j_parent, 2]]), zdir='z')

                        lines_3d[n][j-1][1][0].set_xdata(np.array([pos_gt[j, 0], pos_gt[j_parent, 0]]))
                        lines_3d[n][j-1][1][0].set_ydata(np.array([pos_gt[j, 1], pos_gt[j_parent, 1]]))
                        lines_3d[n][j-1][1][0].set_3d_properties(np.array([pos_gt[j, 2], pos_gt[j_parent, 2]]), zdir='z')

        
        print('{}/{}      '.format(i, limit), end='\r')
        

    fig.tight_layout()
    
    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000/fps, repeat=False)
    if with_club:
        output = f"./outputs/club_{output}"
    else:
        output = f"./outputs/no_club_{output}"

    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=300, writer='imagemagick')
    # elif output.endswith('.pdf'):
    #     for i in range(limit):
    #         # anim.save(f'pdf_res/{i}_ani.pdf', dpi=300, fps=30, writer='matplotlib.animation.PdfWriter')
    #         anim.save(f'pdf_res/{i}_ani.png', dpi=300, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()