import numpy as np


def normalization(img: np.ndarray, shape: str, lock: bool):
    """
    :param img: CHW和HWC均可
    :param shape: 传入CHW或者HWC，输出同一形式
    :param lock: True时，三通道一起归一化，否则分通道归一化
    :return: 返回CHW或HWC
    """

    def compute(data: np.ndarray):
        h, w = data.shape
        data = data.reshape(-1)
        max = np.max(data)
        min = np.min(data)
        if max == min:
            return 65535  # 防止三通道中有一个通道无信息
        else:
            norm_data = (data - min) / (max - min)
            return norm_data.reshape((h, w))

    if lock:
        original_shape = img.shape
        flatten = img.reshape(-1)
        normalized = (flatten - flatten.min()) / (flatten.max() - flatten.min())
        return normalized.reshape(original_shape)
    else:
        if shape == "CHW" or shape == "chw":
            g, r, z = img[0], img[1], img[2]
            norm_g, norm_r, norm_z = compute(g), compute(r), compute(z)
            if type(norm_g) == int or type(norm_r) == int or type(norm_z) == int:
                return 65535
            else:
                return np.array((norm_g, norm_r, norm_z))
        elif shape == "HWC" or shape == "hwc":
            g, r, z = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            norm_g, norm_r, norm_z = compute(g), compute(r), compute(z)
            if type(norm_g) == int or type(norm_r) == int or type(norm_z) == int:
                return 65535
            else:
                return np.concatenate(
                    (norm_g.reshape(256, 256, 1), norm_r.reshape(256, 256, 1), norm_z.reshape(256, 256, 1)), axis=2)
        else:
            raise RuntimeError


def mtf(data: np.ndarray, m: float):
    """
    非线性变换
    方法来源：https://pixinsight.com/doc/tools/HistogramTransformation/HistogramTransformation.html#introduction_002
    :param data: 输入数据
    :param m: midtone值
    :return: 结果
    """
    return ((m - 1) * data) / ((2 * m - 1) * data - m)


def channel_cut(gray):
    """
    单通道进行cut：shadow point， midtones point，higlight point
    :param gray:
    :return:
    """
    highlight = 1.  # 高光处cut，为了保留亮部信息，一般都为1
    hist, bar = np.histogram(gray.reshape(-1), bins=65536)  # 计算16-bit频数直方图
    cdf = hist.cumsum()  # 计算累计分布直方图
    shadow_index = np.argwhere(cdf > 0.001 * gray.reshape(-1).shape[0])[0]  # 计算shadow，最暗0.1%的像素cut掉
    shadow = bar[shadow_index]  # 上一步的索引值来得到实际像素值大小，即shadow大小
    midtones = np.median(gray) - shadow  # 简单计算一个差不多的midtones
    gray[gray < shadow] = shadow    # 小于shadow值的地方置为shadow，后续会进行归一化为0
    gray[gray > highlight] = 1. # 高于highlight值的地方置为highlight
    gray = gray.reshape(-1) # cut之后重新归一化到[0, 1]
    norm_data = (gray - gray.min()) / (gray.max() - gray.min())
    gray = norm_data.reshape((256, 256))
    median = np.median(mtf(gray, midtones)) # 先拉伸一次计算这个时候的median值为多少
    set_median = 1 / 8  # 事先要规定好希望把median拉伸到何处，经试验直方图峰值在左侧1/8处背景黑的合适，亮部也合适
    weight = median / set_median    # 计算之前计算的median值和规定median值差多少比例，得到最终正确的midtones，再去拉伸
    right_midtones = weight * midtones
    return mtf(gray, right_midtones)

def channel_cut1(gray):
    """
    单通道进行cut：shadow point， midtones point，higlight point
    :param gray:
    :return:
    """
    highlight = 1.  # 高光处cut，为了保留亮部信息，一般都为1
    hist, bar = np.histogram(gray.reshape(-1), bins=65536)  # 计算16-bit频数直方图
    cdf = hist.cumsum()  # 计算累计分布直方图
    shadow_index = np.argwhere(cdf > 0.0000001 * gray.reshape(-1).shape[0])[0]  # 计算shadow，最暗0.1%的像素cut掉
    shadow = bar[shadow_index]  # 上一步的索引值来得到实际像素值大小，即shadow大小
    midtones = np.median(gray) - shadow  # 简单计算一个差不多的midtones
    gray[gray < shadow] = shadow    # 小于shadow值的地方置为shadow，后续会进行归一化为0
    gray[gray > highlight] = 1. # 高于highlight值的地方置为highlight
    gray = gray.reshape(-1) # cut之后重新归一化到[0, 1]
    norm_data = (gray - gray.min()) / (gray.max() - gray.min())
    gray = norm_data.reshape((256, 256))
    median = np.median(mtf(gray, midtones)) # 先拉伸一次计算这个时候的median值为多少
    set_median = 1 / 8  # 事先要规定好希望把median拉伸到何处，经试验直方图峰值在左侧1/8处背景黑的合适，亮部也合适
    weight = median / set_median    # 计算之前计算的median值和规定median值差多少比例，得到最终正确的midtones，再去拉伸
    right_midtones = weight * midtones
    return mtf(gray, right_midtones)

def auto_scale(data: np.ndarray):
    g, r, z = channel_cut(data[0]), channel_cut(data[1]), channel_cut(data[2])
    img = np.array((g, r, z))
    img[img < 0] = 0
    img[img > 1] = 1.
    return img
def auto_scale1(data: np.ndarray):
    g, r, z = channel_cut1(data[0]), channel_cut1(data[1]), channel_cut1(data[2])
    img = np.array((g, r, z))
    img[img < 0] = 0
    img[img > 1] = 1.
    return img

def ycl(input_data):
    normalized_unlock = normalization(input_data, shape="CHW", lock=False)  # 先做一次分通道归一化
    if not type(normalized_unlock) == int:
        normalized_unlock = normalization(normalized_unlock, shape="CHW", lock=True)  # 再做一次全通道归一化
        if not type(normalized_unlock) == int:
            normalized_unlock = auto_scale(normalized_unlock)  # 再做拉伸
    return normalized_unlock

def ycl1(input_data):
    normalized_unlock = normalization(input_data, shape="CHW", lock=False)  # 先做一次分通道归一化
    if not type(normalized_unlock) == int:
        normalized_unlock = normalization(normalized_unlock, shape="CHW", lock=True)  # 再做一次全通道归一化
        if not type(normalized_unlock) == int:
            normalized_unlock = auto_scale1(normalized_unlock)  # 再做拉伸
    return normalized_unlock

def ycl2(input_data):
    normalized_unlock = normalization(input_data, shape="CHW", lock=False)  # 先做一次分通道归一化
    return normalized_unlock