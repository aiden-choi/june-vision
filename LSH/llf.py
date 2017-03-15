import numpy as np


# An python implementation for demo for paper "Visual Tracking via Locality Sensitive Histograms"
# by Shengfeng He, Qingxiong Yang, Rynson W.H. Lau, Jiang Wang, and Ming-Hsuan Yang
# To appear in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2013), Portlands, June, 2013.
# feature_img output a (x,y) matrix

class llf(object):
    def __init__(self):
        self.color_max = 256

    def llf(self, img, hist_mtx, k, nbin):
        size_w, size_h = img.shape
        color_range = np.arange(0, self.color_max, self.color_max / nbin)

        # find bin id for each pixel
        bp_mtx = np.zeros((size_w, size_h))
        for i in range(len(color_range) - 1):
            mask = (img >= color_range[i]) & (img < color_range[i + 1])
            bp_mtx[mask] = i
        bp_mtx = np.tile(bp_mtx, (nbin, 1, 1))

        # construct bin id matrix
        b_mtx = np.array([np.ones((size_w, size_h)) * i for i in range(1, nbin + 1)])

        # construct pixel intensity matrix
        i_mtx = np.tile(np.array(img, dtype=np.double), (nbin, 1, 1))

        i_mtx *= k
        i_mtx[i_mtx < k] = k

        # compute illumination invariant features
        x = -(b_mtx - bp_mtx) ** 2 / (2 * i_mtx ** 2)
        e_mtx = np.exp(x)
        temp_e_mtx = np.empty((e_mtx.shape[1], e_mtx.shape[2], e_mtx.shape[0]))

        for i in range(nbin):
            temp_e_mtx[:, :, i] = e_mtx[i, :, :]
        e_mtx = temp_e_mtx

        lp_mtx = e_mtx * hist_mtx
        feature_img = np.sum(lp_mtx, axis=2)

        return feature_img
