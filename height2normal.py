#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
import textwrap
from PIL import Image
import numpy as np
from numpy.lib.stride_tricks import as_strided

if __name__ == '__main__':
    parser = ArgumentParser(description= 'convert height-map to normal-map.',
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('infile', type = str, metavar='FILE',
                        help=textwrap.dedent('''\
                            input file name of height-map PNG image.
                            input image should be grayscale.
                            given rgb image will be converted as grayscale'''))
    parser.add_argument('-n', '--normal', type = str, default = 'normalmap.png',
                        metavar='FILE', dest='normalfile',
                        help=textwrap.dedent("""\
                            output file name of normal-map PNG image.
                            file name extension must be '.png'."""))
    parser.add_argument('-z', '--zangle', type = str, default = 'zanglemap.png',
                        metavar='FILE', dest='zanglemap',
                        help=textwrap.dedent("""\
                            output file name of z-angle-map PNG image.
                            file name extension must be '.png'."""))
    parser.add_argument('-t', '--tapnum', type = int, default = 15,
                        dest='tapnum',
                        help=textwrap.dedent('''\
                            tap num of interpolation filter.
                            it must be 3 or more odd int num.'''))
    parser.add_argument('-r', '--vhratio', type = float, default = 20.,
                        dest='vhratio',
                        help=textwrap.dedent('''\
                            Vertical/Horizontal ratio.
                            it must be positive real num.'''))
    parser.add_argument('-d', '--difference', type = float, default = 1e-10,
                        dest='diff',
                        help=textwrap.dedent('''\
                            difference value of differential.
                            it must be 0.1 or less small positive real num.'''))
    parser.add_argument('-g', '--gamma', type = float, default = 0.1,
                        dest='gamma',
                        help=textwrap.dedent('''\
                            gamma value for z-angle image.
                            it must be positive real num.'''))
    parser.add_argument('-e', '--expand', type = int, default = 0,
                        dest='expand',
                        help=textwrap.dedent('''\
                            Image expand method.
                            0: use opposit side image as tile
                            1: use edge pixel repetition'''))
    parser.add_argument('-b', '--blur', type = int, default = 7,
                        dest='blur',
                        help=textwrap.dedent('''\
                            use blur or not.
                            0: not use
                            3 or more odd int num: use blur
                            when blur is used, num is size of kernel matrix'''))
    parser.add_argument('-s', '--sigma', type = float, default = 1.,
                        dest='sigma',
                        help=textwrap.dedent('''\
                            sigma value for Gaussian Blur.
                            it must be positive real num'''))
    args = parser.parse_args()
    inputfile = args.infile
    outputfile1 = args.normalfile
    outputfile2 = args.zanglemap

    # check parameters
    TAP = args.tapnum
    TAP2 = TAP//2
    if TAP >= 3 and TAP%2 == 1:
        print("interpolation filter TAP: {}".format(TAP))
    else:
        print("invalid TAP num is {}".format(TAP))
        print("--tapnum must be 3 or more odd int num")
        exit()

    KSIZE = args.blur
    SIGMA = args.sigma
    if KSIZE == 0: 
        print("blur filter is not applied")
    elif KSIZE >= 3 and KSIZE%2 == 1:
        print("kernel matrix size for blur: {}x{}".format(KSIZE, KSIZE))
        if SIGMA > 0.:
            print("sigma value of Gaussian Blur: {}".format(SIGMA))
        else:
            print("invalid sigma value of Gaussian Blur: {}".format(SIGMA))
            print("--sigma value must be positive real num")
            exit()
    else:
        print("invalid blur kernel matrix num: {}".format(KSIZE))
        print("--blur must be 3 or more odd int num if it's requierd")
        print("or not requierd, it must be 0 int num")
        exit()

    D = args.diff
    if D > 0. and D <= 0.1:
        print("difference num: {}".format(D))
    else:
        print("invalid num of Difference num {}".format(D))
        print("--diff must be 0.1 or less small positive real num")
        exit()
    LT128 = 2.**7.-D
    LT65536 = 2.**16.-D

    GAMMA = args.gamma
    if GAMMA > 0.:
        print("gamma value for z-angle image: {}".format(GAMMA))
    else:
        print("invalid num of gamma value {}".format(GAMMA))
        print("--gamma must be positive real num")
        exit()

    EMETHOD = args.expand
    if EMETHOD != 1 and EMETHOD !=0:
        print("invalid num of Image expand method:{}".format(EMETHOD))
        print("--expand must be 0 or 1 int num")
        exit()

    VHRATIO = args.vhratio
    if VHRATIO > 0.:
        print("Vertical/Horizontal Ratio:", VHRATIO)
    else:
        print("invalid num of V/H Ratio {}".format(VHRATIO))
        print("--vhratio must be positive real num")
        exit()

    # then load image and check it
    image = Image.open(inputfile)
    if image.format != 'PNG':
        print("input image must be PNG")
        exit()
    print("input image format is", image.format)

    if image.mode == 'I':
        print("input image mode is 16bit grayscale")
        height = np.array(image, np.float)
    elif image.mode == 'L':
        print("input image mode is 8bit grayscale")
        height = np.array(image, np.float)*2.**8.
    elif image.mode == '1':
        print("input image mode is 1bit black/white")
        height = np.array(image, np.float)*2.**16.
    elif image.mode == 'RGB' or image.mode == 'RGBA':
        # convert rgb image to grayscale using ITU-R Rec BT.601
        tmp = np.array(image, np.float)
        R = 0.299
        G = 0.587
        B = 0.114
        height = (tmp[:, :, 0]*R+tmp[:, :, 1]*G+tmp[:, :, 2]*B)*2.**8.
        print("input image mode is 24bit color")
        print("convert to grayscale")
    else:
        print("not supported image mode type:{}".format(image.mode))
        exit()
    print("input image size: {}".format(image.size))
    print("input image value range: {}".format(image.getextrema()))

    if image.mode != 'I':
        print("expand to 16 bit depth")

    height = height/VHRATIO

    # Expand image for TAPs
    EXP = TAP2+KSIZE//2
    if EMETHOD == 1:
        print("enlarge image with edge pixel repetition.")
        # for x axis
        exp1 = np.tile(height[:, [0]], (1, EXP))
        exp2 = np.tile(height[:, [-1]], (1, EXP))
        height = np.concatenate([exp1, height, exp2], 1)

        # for y axis
        exp1 = np.tile(height[[0], :], (EXP, 1))
        exp2 = np.tile(height[[-1], :], (EXP, 1))
        height = np.concatenate([exp1, height, exp2], 0)
    else:
        print("enlarge image with opposit side image as tile.")
        # for x axis
        exp1 = height[:, -1-EXP:-1]
        exp2 = height[:, 0:EXP]
        height = np.concatenate([exp1, height, exp2], 1)

        # for y axis
        exp1 = height[-1-EXP:-1, :]
        exp2 = height[0:EXP, :]
        height = np.concatenate([exp1, height, exp2], 0)

    # applay Gaussian Blur filter if it's requierd
    if KSIZE != 0:
        # make Gaussian Blur kernel matrix
        vec = np.linspace(-float(KSIZE//2), float(KSIZE//2), KSIZE)
        xx, yy = np.meshgrid(vec, vec)
        kernel = np.exp(-0.5*(xx**2.+yy**2.)/SIGMA**2.)
        kernel /= np.sum(kernel)

        # convolve each pixels with Gaussian Blur kernel matrix
        new_shape = tuple(np.subtract(height.shape, kernel.shape)+1)
        view = as_strided(height, kernel.shape+new_shape, height.strides*2)
        height = np.einsum('ij,ijkl->kl', kernel, view)

    # make filter vector for interpolation as truncated sinc
    Vl = np.linspace(float(-TAP2), float(TAP2), TAP2*2+1)
    Vsn = np.sinc(Vl-D) # for negative nearest neighbor point
    Vsp = np.sinc(Vl+D) # for positive nearest neighbor point

    # make initialization vector for xyz vector arary elements 
    rsz = np.zeros((height.shape[0]-TAP2*2, height.shape[1]-TAP2*2))
    rsd = np.full((height.shape[0]-TAP2*2, height.shape[1]-TAP2*2), D*2)

    # Differentiate the heightmap and make xyz vector array
    #  for y-axis
    rsp_y = rsz.T.copy()
    rsn_y = rsz.T.copy()
    for x in range(height.shape[1]-TAP2*2):
        rsp_y[x] = np.convolve(height.T[x+TAP2], Vsp, mode='valid')
        rsn_y[x] = np.convolve(height.T[x+TAP2], Vsn, mode='valid')
    xyz_dy = np.stack([rsz, rsd, rsn_y.T-rsp_y.T], axis=-1)

    #  for x-axis
    rsp_x = rsz.copy()
    rsn_x = rsz.copy()
    for y in range(height.shape[0]-TAP2*2):
        rsp_x[y] = np.convolve(height[y+TAP2], Vsp, mode='valid')
        rsn_x[y] = np.convolve(height[y+TAP2], Vsn, mode='valid')
    xyz_dx = np.stack([rsd, rsz, rsp_x-rsn_x], axis=-1)

    # make normal-map image and z-angle-map image
    cross = np.cross(xyz_dx, xyz_dy)
    norm = np.linalg.norm(cross, axis=-1)
    normal = cross/np.stack([norm, norm, norm], axis=-1)
    normalmap = normal*LT128+127.
    zanglemap = (1.-np.abs(normal[:, :, 2]))**(1./GAMMA)*LT65536

    # save normal-map image
    Image.fromarray(normalmap.astype(np.uint8)).save(outputfile1)
    print("output normal map file name is", outputfile1)

    # save z-angle-map image
    Image.fromarray(zanglemap.astype(np.uint16)).save(outputfile2)
    print("output z-angle map file name is", outputfile2)

    print("Done!")

