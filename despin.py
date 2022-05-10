from pathlib import Path
from functools import partial
import multiprocessing
import subprocess
from operator import attrgetter
from typing import NamedTuple
from enum import Enum
import pickle
import tempfile

import matplotlib.pyplot as plt

import numpy as np

import skimage
from skimage.color import rgb2gray, gray2rgb
from skimage.feature import match_descriptors, SIFT
import skimage.io
from skimage.io import imread
from skimage.transform import EuclideanTransform
from skimage.util import compare_images
from skimage import measure

CACHE = True
TMP = Path(tempfile.gettempdir())

class Features(NamedTuple):
    frame_number: int
    frame_file: Path
    keypoints: np.array   # n x 2
    descriptors: np.array # n x m

class Transform(NamedTuple):
    a_frame_number: int
    b_frame_number: int
    matches: np.array # n x 2
    mat: np.array # 4x4

class FramePairingMethod(Enum):
    first = 'first'
    consecutive = 'consecutive'

imsave = partial(skimage.io.imsave, compress_level=3, plugin='pil')

def frames_to_video(images, outname, fps=30):
    args = [
        'ffmpeg',
        '-f', 'image2pipe',
        '-vcodec', 'png',
        '-r', str(fps),
        '-i', '-',
        '-vcodec', 'png',
        '-q:v', '0',
        '-y',
        outname,
    ]

    pipe = subprocess.Popen(args, stdin=subprocess.PIPE)

    for im in images:
        imsave(pipe.stdin, im)

    pipe.stdin.close()
    pipe.wait()

    # Make sure all went well
    if pipe.returncode != 0:
        raise subprocess.CalledProcessError(pipe.returncode, args)

def frame_number(x):
    return int(Path(x).stem)

def pickle_load(fname):
    with open(fname, 'rb') as fh:
        return pickle.load(fh)

def pickle_save(fname, data):
    with open(fname, 'wb') as fh:
        return pickle.dump(data, fh)

def kmers(xs, k):
    for i in range(len(xs) - k + 1):
        yield xs[i:i+k]
pairs = partial(kmers, k=2)

def img_corners(im):
    h, w = im.shape[:2]
    # top-left, top-right, bottom-right, bottom-left
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]])

def scale_frame(frame, scale=4, anti_aliasing=True):
    if scale == 1:
        return frame

    if frame.dtype == bool:
        anti_aliasing = False

    h, w = frame.shape[:2]
    return skimage.transform.resize(frame, (h // scale, w // scale), anti_aliasing=anti_aliasing)

def prep_image(frame_file, scale=4):
    return scale_frame(rgb2gray(imread(frame_file)), scale=scale)

def video_to_frames(filename: Path, outdir: Path, pattern='%04d.png'):
    if not outdir.exists():
        outdir.mkdir()
        subprocess.run(['ffmpeg', '-i', str(filename), str(outdir / pattern)], check=True)

    return sorted(outdir.glob('*.png'))

# we have a 4k video so upsampling is pointless
def get_features(frame_file, scale=1, upsampling=1, mask=None, cache=CACHE):
    outname = frame_file.with_suffix('.features')

    if cache and outname.exists():
        return pickle_load(outname)

    im = prep_image(frame_file, scale=scale)

    d = SIFT(upsampling=upsampling)

    d.detect_and_extract(im)

    if mask is not None:
        keep = mask[d.keypoints[:, 0], d.keypoints[:, 1]]
        d.keypoints = d.keypoints[keep]
        d.descriptors = d.descriptors[keep]

    ret = Features(
        frame_number = frame_number(frame_file),
        frame_file  = frame_file,
        keypoints   = d.keypoints,
        descriptors = d.descriptors,
    )

    if cache:
        pickle_save(outname, ret)

    return ret

def get_transform(a: Features, b: Features, min_samples=4, residual_threshold=0.5, cache=CACHE):
    outname = a.frame_file.parent / f'{a.frame_number:04d}-{b.frame_number:04d}.transform'
    if cache and outname.exists():
        return pickle_load(outname)

    # flip the keypoints from (row, col) to (col, row) as this matches the (x, y) convention
    # in the solvers
    # see https://github.com/scikit-image/scikit-image/issues/1749
    kp1 = np.flip(a.keypoints, axis=-1)
    kp2 = np.flip(b.keypoints, axis=-1)

    matches = match_descriptors(a.descriptors, b.descriptors, max_ratio=0.5, cross_check=True)

    kp1 = kp1[matches[:, 0]]
    kp2 = kp2[matches[:, 1]]

    mat, inliers = measure.ransac((kp2, kp1), EuclideanTransform, min_samples=min_samples, residual_threshold=residual_threshold)

    ret = Transform(
        a_frame_number = a.frame_number,
        b_frame_number = b.frame_number,
        matches = matches,
        mat = mat,
    )

    if cache:
        pickle_save(outname, ret)

    return ret

def show_keypoints(im, kp, color=(0, 1, 0), matches=None, matches_color=(1, 0, 0)):
    if len(im.shape) == 2:
        im = gray2rgb(im)
    else:
        im = im.copy()

    im[kp[:, 0], kp[:, 1]] = color

    if matches is not None:
        kpm = kp[matches]
        im[kpm[:, 0], kpm[:, 1]] = matches_color

    return im

def get_all_features(frame_files, mask, scale):
    get_features_ = partial(get_features, scale=scale, mask=mask)

    with multiprocessing.Pool() as p:
        ret = sorted(p.imap_unordered(get_features_, frame_files), key=attrgetter('frame_number'))

    return ret

def get_transform_worker(t):
    return get_transform(*t)

def get_all_transforms(features, method=FramePairingMethod.consecutive):
    if method == FramePairingMethod.consecutive:
        args = pairs(features)
    elif method == FramePairingMethod.first:
        args = ((features[0], f) for f in features[1:])

    with multiprocessing.Pool() as p:
        ret = sorted(p.imap_unordered(get_transform_worker, args), key=attrgetter('a_frame_number', 'b_frame_number'))

    return ret

def plot_transforms(transforms, outname, accumulate=True):
    fig, ax = plt.subplots(figsize=(12, 6), nrows=3, ncols=2, squeeze=False)

    id = lambda x: x
    maybe_cumsum = np.cumsum if accumulate else id

    rotations = np.rad2deg([t.mat.rotation for t in transforms])
    t_xs = [t.mat.translation[0] for t in transforms]
    t_ys = [t.mat.translation[1] for t in transforms]

    ax[0][0].set_title('Δθ (deg)')
    ax[0][0].plot(rotations)
    ax[0][1].set_title('θ (deg)')
    ax[0][1].plot(maybe_cumsum(np.rad2deg(rotations)))

    ax[1][0].set_title('ΔX')
    ax[1][0].plot(t_xs)
    ax[1][1].set_title('X')
    ax[1][1].plot(maybe_cumsum(t_xs))

    ax[2][0].set_title('ΔY')
    ax[2][0].plot(t_ys)
    ax[2][1].set_title('Y')
    ax[2][1].plot(maybe_cumsum(t_ys))

    plt.tight_layout()
    plt.savefig(outname)

def plot_transforms_whole_space(corners, transforms, outname, accumulate=True):
    plt.figure(figsize=(6, 6))
    plt.gca().set_aspect('equal')

    def draw_corners(corners, color='r', alpha=0.1):
        x = corners[:, 0]
        y = corners[:, 1]
        plt.plot(x, y, alpha=alpha, color=color)
        plt.plot((x[0], x[-1]), (y[0], y[-1]), alpha=alpha, color=color)

    orig_corners = corners.copy()
    cur = corners

    draw_corners(corners, color='g', alpha=1)
    for t in transforms:
        if accumulate:
            cur = t.mat.inverse(cur)
        else:
            cur = t.mat.inverse(orig_corners)
        draw_corners(cur)
    plt.tight_layout()
    plt.savefig(outname)

def transform_frames(frame_files, transforms, scale=1, accumulate=True):
    yield scale_frame(imread(frame_files[0]), scale=scale)

    cur = EuclideanTransform()

    for frame_file, transform in zip(frame_files[1:], transforms):
        if accumulate:
            cur = cur + transform.mat
        else:
            cur = transform.mat

        im = scale_frame(imread(frame_file), scale=scale)
        yield skimage.transform.warp(im, cur.inverse)

def make_video(frame_files, transforms, outname, scale=1, accumulate=True):
    frames_to_video(transform_frames(frame_files, transforms, scale=scale, accumulate=accumulate), outname)

# Len 2
def make_matching_frame(frames, features, transform, outname):
    i1, i2 = frames
    f1, f2 = features
    i1 = show_keypoints(i1, f1.keypoints, matches=transform.matches[:, 0])
    i2 = show_keypoints(i1, f1.keypoints, matches=transform.matches[:, 1])

    together = np.hstack([i1, i2])
    imsave(outname, together)

def make_blended_frame(frames, transform, outname):
    i1, i2 = frames
    mat = transform.mat

    imsave(outname, compare_images(i1, skimage.transform.warp(i2, mat.inverse), method='blend'))

def main(args):
    frame_files = video_to_frames(args.video, TMP / f'despin-{args.video.stem}.frames')
    # -1 b/c frames are counted from 1
    frame_files = frame_files[args.start_frame - 1: args.end_frame]

    if args.start_frame != frame_number(frame_files[0]):
        print(f'ERROR expected start_frame={args.start_frame} to match, but got {frame_number(frame_files[0])}')
    if args.end_frame != frame_number(frame_files[-1]):
        print(f'ERROR expected end_frame={args.end_frame} to match, but got {frame_number(frame_files[-1])}')

    mask = None
    if args.mask:
        mask = scale_frame(imread(args.mask, as_gray=True) > 0, args.scale)

    features = get_all_features(frame_files, mask=mask, scale=args.scale)
    transforms = get_all_transforms(features)

    prep_image_ = partial(prep_image, scale=args.scale)
    first_two = list(map(prep_image_, frame_files[:2]))
    fn = args.start_frame

    make_matching_frame(first_two, features[:2], transforms[0], TMP / f'despin-matching-{fn:04d}-{fn+1:04d}.png')
    make_blended_frame(first_two, transforms[0], TMP / f'despin-blended-{fn:04d}-{fn+1:04d}.png')

    accumulate = args.method == FramePairingMethod.consecutive

    plot_transforms(transforms, TMP / f'despin-transforms-{args.method.value}.png', accumulate=accumulate)
    plot_transforms_whole_space(img_corners(first_two[0]), transforms, TMP / f'despin-transforms-whole-space-{args.method.value}.png', accumulate=accumulate)

    make_video(frame_files, transforms, TMP / f'{args.video.stem}-despin-{args.method.value}.mp4', scale=args.scale, accumulate=accumulate)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    parser.add_argument('--start-frame', type=int, required=True, help='frames are counted from 1')
    parser.add_argument('--end-frame', type=int, required=True)
    parser.add_argument('--mask', required=False)
    parser.add_argument('--scale', default=4, type=float, help='power of 2, reciprocal scaling (1 / 4 default)')
    parser.add_argument('--method', default='consecutive', help='consecutive|first', type=FramePairingMethod)
    args = parser.parse_args()

    main(args)
