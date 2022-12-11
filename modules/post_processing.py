import cv2
import math
import glob
import numpy as np
from PIL import Image

RATIO = 1.6


def get_kernel_info(kernel_file):
    input_data = [x.strip() for x in open(kernel_file, 'r').readlines()][0].split(' ')
    k = int(input_data[0])
    sigma = float(input_data[1])
    return k, sigma


def get_gaussian_kernal(k, sigma):
    pi = math.pi
    exp = math.exp
    size = 2*k+1
    g = lambda i, j: (1/(2 * pi * sigma**2)) * exp(-( i**2 + j**2 )/(2 * sigma**2))

    kernel = np.zeros((size,size))
    for i in range(-k,k+1,1):
        for j in range(-k,k+1,1):
            kernel[i+k,j+k] = g(i,j)
    return kernel


def post_process_high_contrast(img):
    return 1 - cv2.equalizeHist(img)


def high_contrast(input_path=None, output_path=None):
    
    in_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)#.astype(np.uint8)
    out_image = 1 - cv2.equalizeHist(in_image)
    Image.fromarray(np.uint8(out_image * 255)).save(output_path)


def high_contrast2(input_path=None, output_path=None):

    in_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)#.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
    out_image = 1 - clahe.apply(in_image)
    Image.fromarray(np.uint8(out_image * 255)).save(output_path)


# def apply_image_contrast(in_image):
#     print(in_image)
#     print('Converting image to LAB Color model')
#     lab = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
#     print('Splitting the LAB image to different channels')
#     l, a, b = cv2.split(lab)
#     print('Applying CLAHE')
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     cl = clahe.apply(l)
#     print('Merge')
#     out_image = cv2.merge((cl, a, b))
#     return out_image


def post_process_laplacian(img):
    kernel_path = 'modules/kernel_info/4.txt'
    k, sigma = get_kernel_info(kernel_path)
    kernel = get_gaussian_kernal(k, sigma) - get_gaussian_kernal(k, sigma*RATIO)
    dog_img = apply_DoG(img, kernel)
    return dog_img


def difference_of_gaussian(input_path=None, output_path=None, kernel_path=None):
    kernel_path = 'modules/kernel_info/3.txt'

    in_image = np.array(Image.open(input_path).convert('RGB')) / 255
    k, sigma = get_kernel_info(kernel_path)
    kernel = get_gaussian_kernal(k, sigma) - get_gaussian_kernal(k, sigma*RATIO)
    out_image = apply_DoG(in_image, kernel)
    Image.fromarray(np.uint8(out_image * 255)).save(output_path)


def apply_DoG(in_image, kernel):
    out_image = np.zeros(shape=in_image.shape)
    for i in range(3):
        out_image[:,:,i] = cv2.filter2D(in_image[:,:,i], -1, kernel)
    return 1 - out_image


# def testing():
#     # img = cv2.imread('../test/image/1.jpg')
#     img = np.array(Image.open('test/image/ProgrammingLanguage.jpg').convert('RGB')) / 255
#     # img = np.array(Image.open('test/result/gradient/1.jpg').convert('L')) / 255
#     gray = img[:,:,0]

#     mask1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
#     mask2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
#     mask3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) # sharpening
#     laplacian1 = cv2.filter2D(gray, -1, mask1)
#     laplacian2 = cv2.filter2D(gray, -1, mask2)
#     laplacian3 = cv2.filter2D(gray, -1, mask3)
#     laplacian4 = cv2.Laplacian(gray, -1)

#     # Image.fromarray(np.uint8(gray)).save(f'test/post_processing_result/tmp/original0.jpg')
#     Image.fromarray(np.uint8(gray * 255)).save(f'test/post_processing_result/tmp/original.jpg')
#     Image.fromarray(np.uint8(laplacian1 * 255)).save(f'test/post_processing_result/tmp/laplacian11.jpg')
#     Image.fromarray(np.uint8(laplacian2 * 255)).save(f'test/post_processing_result/tmp/laplacian12.jpg')
#     Image.fromarray(np.uint8(laplacian3 * 255)).save(f'test/post_processing_result/tmp/laplacian13.jpg')
#     Image.fromarray(np.uint8(laplacian4 * 255)).save(f'test/post_processing_result/tmp/laplacian14.jpg')

#     laplacian1 = 1 - laplacian1
#     laplacian2 = 1 - laplacian2
#     laplacian3 = 1 - laplacian3
#     laplacian4 = 1 - laplacian4

#     Image.fromarray(np.uint8(laplacian1 * 255)).save(f'test/post_processing_result/tmp/laplacian21.jpg')
#     Image.fromarray(np.uint8(laplacian2 * 255)).save(f'test/post_processing_result/tmp/laplacian22.jpg')
#     Image.fromarray(np.uint8(laplacian3 * 255)).save(f'test/post_processing_result/tmp/laplacian23.jpg')
#     Image.fromarray(np.uint8(laplacian4 * 255)).save(f'test/post_processing_result/tmp/laplacian24.jpg')

#     #cv2.waitKey(0)


# input_path = 'test/image/ProgrammingLanguage.jpg'
# output_path = f'test/post_processing_result/tmp/ProgrammingLanguage_highcontrast.jpg'
# input_path = 'test/result/gradient/ProgrammingLanguage.jpg'
# output_path = f'test/post_processing_result/tmp/ProgrammingLanguage_cropped_high.jpg'
# input_path = 'test/result/gradient/1.jpg'
# output_path = f'test/post_processing_result/tmp/1_cropped_high.jpg'


def main():
    for image_path in glob.glob('test/image/*'):
        for warp_type in ['simple', 'ratio_given', 'gradient']:
            input_path = image_path.replace('image', f'result/{warp_type}'); print(input_path)
            # high contrast
            output_path = image_path.replace('image', f'final_result/high_contrast/{warp_type}'); print(output_path)
            high_contrast2(input_path, output_path)
            # laplacian
            output_path = image_path.replace('image', f'final_result/laplacian/{warp_type}'); print(output_path)
            difference_of_gaussian(input_path, output_path)
        
if __name__ == '__main__':
    #testing()
    #high_contrast()
    #high_contrast2()
    #difference_of_gaussian()
    main()