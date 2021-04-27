"""
AiueoABC Temperature Record System (Atres)
エイターズ
"""
import cv2
import numpy as np


class atres:
    """
    int16KelvinArray => numpy array using uint16, for exaple, each
                        data in array looks like 27315 at 273.15 K
    save => to check
    atresimg => this will contain temperature data as rgb data
    """
    def temp2color(self, int16KelvinArray, save=True):
        bStack, gStack = np.uint8(divmod(int16KelvinArray, 256))
        rStack = __raw_to_8bit(int16KelvinArray)
        atresimg = cv2.merge((bStack,gStack,rStack))
        if save:
            cv2.imwrite("colorTemperatureData.png", atresimg)
        return atresimg

    def bodytemp2color(self, int16KelvinArray, save=True):
        bStack, gStack = np.uint8(divmod(int16KelvinArray, 256))
        rStack = __raw_to_8bit_body(int16KelvinArray)
        atresimg = cv2.merge((bStack, gStack, rStack))
        if save:
            cv2.imwrite("AtresDataForBody.png", atresimg)
        return atresimg

    def rawtemp2color(self, int16KelvinArray, save=True):
        bStack, gStack = np.uint8(divmod(int16KelvinArray, 256))
        rStack = bStack
        atresimg = cv2.merge((bStack, gStack, rStack))
        if save:
            cv2.imwrite("AtresDataRAW.png", atresimg)
        return atresimg

    """
    atresimg => this should be the img come through above functions
    """
    def atresimg2temp(self, atresimg):
        upperStack, lowerStack, _ = cv2.split(atresimg)
        int16KelvinArray = np.uint16(upperStack * 256 + lowerStack)
        return int16KelvinArray


def __raw_to_8bit(data):
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return np.uint8(data)


def __raw_to_8bit_body(data):
    data = (np.clip(data, 29300, 31800) - 29300) / 2500 * 255
    # data = np.uint16(np.clip(data, 29300, 31800) * 2)
    # np.right_shift(data, 8, data)
    return np.uint8(data)

