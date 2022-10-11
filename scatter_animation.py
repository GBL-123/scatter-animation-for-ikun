import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numba import jit


def get_gaussian_kernel(sigma, size):
    '''定义高斯滤波器，size只能是奇数'''
    
    if size % 2 == 0:
        raise ValueError("size只能是奇数")
    
    idx = np.arange(-(size - 1)/2, (size - 1)/2 + 1)
    x, y = np.meshgrid(idx, idx, indexing="ij")
    kernel = 1/(2*np.pi*sigma**2)*np.exp(-(x**2 + y**2)/(2*sigma**2))
    kernel_matrix = kernel/np.sum(kernel)
    
    return kernel_matrix


@jit(nopython=True)
def gaussian_filter(gaussian_kernel, gray_img):
    '''进行高斯滤波处理'''
    
    filter_img = gray_img.copy()
    a = int((gaussian_kernel.shape[0] - 1)/2)
    for i in range(a, gray_img.shape[0] - a):
        for j in range(a, gray_img.shape[1] - a):
            matrix = gray_img[i - a:i + a + 1, j - a:j + a + 1]
            filter_img[i,j] = np.sum(matrix*gaussian_kernel)
    filter_img = filter_img.astype("uint8")

    return filter_img


@jit(nopython=True)
def adjust_direction(direction):
    '''将梯度方向近似为某个角度'''
    
    angels = np.linspace(-np.pi/2, np.pi/2, 5)
    adj_direction = angels[np.argmin(np.abs(direction - angels))]
    
    return adj_direction


@jit(nopython=True)
def calculate_grad(filter_img):
    '''计算梯度值和梯度方向'''
    
    sobel_matrix = np.array([[1,0,-1], [2,0,-2], [1,0,-1]]) # sobel算子
    
    grad_matrix = filter_img.copy()
    direction_matrix = np.zeros(filter_img.shape)
    for i in range(1, filter_img.shape[0] - 1):
        for j in range(1, filter_img.shape[1] - 1):    
            dx = np.sum(filter_img[i - 1:i + 2, j - 1:j + 2]*sobel_matrix)
            dy = np.sum(filter_img[i - 1:i + 2, j - 1:j + 2]*sobel_matrix.T)
            grad = np.sqrt(dx**2 + dy**2)
            if dx == 0:
                dx = dx + 0.01
            direction = np.arctan(dy/dx)
            adj_direction = adjust_direction(direction) # 将梯度方向近似成某一个角度
            grad_matrix[i,j] = grad
            direction_matrix[i,j] = adj_direction
    grad_matrix = grad_matrix.astype("uint8")
    
    return grad_matrix, direction_matrix


@jit(nopython=True)
def NMS(grad_matrix, direction_matrix):
    '''非极大值抑制'''
    
    adj_grad_matrix = np.zeros(grad_matrix.shape)
    for i in range(1, grad_matrix.shape[0] - 1):
        for j in range(1, grad_matrix.shape[1] - 1):
        
            if (np.abs(direction_matrix[i,j]) == np.pi/2) | (np.abs(direction_matrix[i,j]) == -np.pi/2): # 选梯度的比较方向
                if grad_matrix[i,j] == np.max(grad_matrix[i - 1:i + 2,j]):
                    adj_grad_matrix[i,j] = grad_matrix[i,j]
            
            elif direction_matrix[i,j] == 0:
                if grad_matrix[i,j] == np.max(grad_matrix[i,j - 1:j + 2]):
                    adj_grad_matrix[i,j] = grad_matrix[i,j]
            
            elif direction_matrix[i,j] == np.pi/4:
                if grad_matrix[i,j] == np.max(
                        np.diag(np.fliplr(grad_matrix[i - 1:i + 2, j - 1:j + 2]))
                        ):
                    adj_grad_matrix[i,j] = grad_matrix[i,j]
            
            elif direction_matrix[i,j] == -np.pi/4:
                if grad_matrix[i,j] == np.max(
                        np.diag(grad_matrix[i - 1:i + 2, j - 1:j + 2])
                        ):
                    adj_grad_matrix[i,j] = grad_matrix[i,j]
    
    return adj_grad_matrix


def get_threshold(img):
    '''利用图像的标准差确定双阈值，如果用其他策略可重写该函数'''
    
    threshold1 = np.std(img)*2
    threshold2 = np.std(img)*3
    
    return threshold1, threshold2


@jit(nopython=True)
def get_outline(adj_grad_matrix, threshold1, threshold2):
    '''根据梯度矩阵进行双阈值筛选'''
    
    a = min(threshold1, threshold2)
    b = max(threshold1, threshold2)
    
    outline_matrix = np.zeros(adj_grad_matrix.shape)
    for i in range(1, adj_grad_matrix.shape[0] - 1):
        for j in range(1, adj_grad_matrix.shape[1] - 1):
        
            if adj_grad_matrix[i,j] > b:
                outline_matrix[i,j] = 255
            
            elif adj_grad_matrix[i,j] >= a:
                if np.max(adj_grad_matrix[i - 1:i + 2, j - 1:j + 2]) > b:
                    outline_matrix[i,j] = 255
    
    return outline_matrix


def get_img_outline(gaussian_kernel, gray_img):
    '''获取灰度图像的轮廓，使用canny算法'''
    
    # 高斯滤波
    filter_img = gaussian_filter(gaussian_kernel, gray_img)
    
    # 计算梯度和梯度方向
    grad_matrix, direction_matrix = calculate_grad(filter_img)
    
    # 非极大值抑制
    adj_grad_matrix = NMS(grad_matrix, direction_matrix)
    
    # 双阈值筛选边缘
    threshold1, threshold2 = get_threshold(adj_grad_matrix)
    img_outline = get_outline(adj_grad_matrix, threshold1, threshold2)
    
    return img_outline


def get_ouline_data(img_outline):
    '''获取轮廓像素的坐标'''
    
    idx = np.argwhere(img_outline==255)

    coordinate_data = idx.copy()
    coordinate_data[:,0] = idx[:,1]
    coordinate_data[:,1] = img_outline.shape[0] - 1 - idx[:,0]
    
    return coordinate_data


def make_animation(img_data):
    '''根据每一帧的坐标数据制作动画'''
    
    fig, ax = plt.subplots(figsize=(16, 9))
    line, = ax.plot([], [], "o", ms=1, c="black")
    ax.set_xlim(0, 1279)
    ax.set_ylim(0, 719)

    def init():
        line.set_data([],[])
        return line,

    def update(n):
        x = img_data[n][:,0]
        y = img_data[n][:,1]
        ax.set_xlim(0, 1279)
        ax.set_ylim(0, 719)
        line.set_data(x, y)
        return line,
        
    ani = FuncAnimation(
        fig, 
        update, 
        list(range(len(img_data))), 
        interval=1/25*1000, 
        init_func=init, 
        blit=True
        )
    
    ani.save("animation.mp4", fps=25)


def main():
    
    video_path = "素材/video.mp4"
    video = cv.VideoCapture(video_path) # 读取视频
    img_data = [] # 用来存放每一张图片的轮廓数据
    
    gaussian_kernel = get_gaussian_kernel(1, 5)
    
    i = 1
    while True: # 逐帧提取图片的轮廓数据，用来画散点图
        
        
        ret, img = video.read()
        
        if ret == False:
            break

        print("正在解析第{}帧图像...".format(i))

        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)        
        img_outline = get_img_outline(gaussian_kernel, gray_img) 
        coordinate_data = get_ouline_data(img_outline)
        img_data.append(coordinate_data)
        i += 1

    print("正在生成动画...")
    make_animation(img_data)
    input("动画生成成功，按回车键退出。")


if __name__ == "__main__":
    main()



