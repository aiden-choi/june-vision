import numpy as np

# An python implementation for demo for paper "Visual Tracking via Locality Sensitive Histograms"
# by Shengfeng He, Qingxiong Yang, Rynson W.H. Lau, Jiang Wang, and Ming-Hsuan Yang
# To appear in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2013), Portlands, June, 2013.
# hist_mtx output a (x,y,b) matrix

class lsh(object):
    def __init__(self):
        self.color_max = 255

    def lsh(self, img, sigma, nbin):
        size_w, size_h = img.shape
        color_range = np.arange(0, self.color_max, self.color_max / nbin)

        # alpha
        alpha_x = np.exp(-np.sqrt(2) / (sigma * size_w))
        alpha_y = np.exp(-np.sqrt(2) / (sigma * size_h))

        # compute Q
        q_mtx = np.zeros(size_w * size_h * nbin).reshape((size_w, size_h, nbin))
        for i in range(len(color_range) - 1):
            mask = (img >= color_range[i]) & (img < color_range[i + 1])
            q_mtx[:, :, i] = np.array(mask, q_mtx.dtype)

        # compute H and normalization factor
        hist_mtx = q_mtx
        f_mtx = np.ones(q_mtx.shape)

        # -----------x dimension
        # compute left part
        hist_mtx_l = hist_mtx.copy()
        f_mtx_l = f_mtx.copy()
        for i in range(1, hist_mtx.shape[0]):
            hist_mtx_l[i, :, :] = hist_mtx_l[i, :, :] + alpha_x * hist_mtx_l[i - 1, :, :]
            f_mtx_l[i, :, :] = f_mtx_l[i, :, :] + alpha_x * f_mtx_l[i - 1, :, :]
        # compute right part
        hist_mtx_r = hist_mtx.copy()
        f_mtx_r = f_mtx.copy()
        for i in range((hist_mtx.shape[0] - 1) - 1, 0, -1):
            hist_mtx_r[i, :, :] = hist_mtx_r[i, :, :] + alpha_x * hist_mtx_r[i + 1, :, :]
            f_mtx_r[i, :, :] = f_mtx_r[i, :, :] + alpha_x * f_mtx_r[i + 1, :, :]
        hist_mtx = hist_mtx_r + hist_mtx_l - q_mtx
        f_mtx = f_mtx_r + f_mtx_l - 1

        # -----------y dimension
        # compute left part
        hist_mtx_l = hist_mtx.copy()
        f_mtx_l = f_mtx.copy()
        for i in range(1, hist_mtx.shape[1]):
            hist_mtx_l[:, i, :] = hist_mtx_l[:, i, :] + alpha_y * hist_mtx_l[:, i - 1, :]
            f_mtx_l[:, i, :] = f_mtx_l[:, i, :] + alpha_x * f_mtx_l[:, i - 1, :]
        # compute right part
        hist_mtx_r = hist_mtx.copy()
        f_mtx_r = f_mtx.copy()
        for i in range((hist_mtx.shape[1] - 1) - 1, 0, -1):
            hist_mtx_r[:, i, :] = hist_mtx_r[:, i, :] + alpha_y * hist_mtx_r[:, i + 1, :]
            f_mtx_r[:, i, :] = f_mtx_r[:, i, :] + alpha_x * f_mtx_r[:, i + 1, :]
        hist_mtx = hist_mtx_r + hist_mtx_l - q_mtx
        f_mtx = f_mtx_r + f_mtx_l - 1

        hist_mtx = hist_mtx / f_mtx

        return hist_mtx
