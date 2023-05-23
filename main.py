import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import cv2
import sys
import scipy
import skimage
from skimage.util import img_as_float
from skimage.filters import sobel
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import median_filter

###############################
def normalizeRange(Img, minVal, maxVal):
    """rescales the image values to a range of 0 to 1 by subtracting the minimum value
    and dividing by the dynamic range. Finally, it scales the image values to the specified
    output range (minVal to maxVal) by multiplying by the output range and adding the minimum value.
    The function returns the normalized image as a numpy array."""

    dynamic_range_s = np.nanmax(Img) - np.nanmin(Img)
    dynamic_range_d = maxVal - minVal

    Out = np.divide((Img - np.nanmin(Img)), dynamic_range_s, out=np.zeros_like(Img),
                    where=dynamic_range_s != 0) * dynamic_range_d + minVal

    return Out
######################################
def sneak(image, inital_contour, alpha=0.01, beta=0.1,
          w_line=2, w_edge=1, tau=100
          , max_px_move=1.0,
          max_iter=2000):
    # preprocessing steps
    contour_history = {}
    # Gaussian smoothing
    img = img_as_float(image)
    img = median_filter(img, size=3)
    img = normalizeRange(img, 0, 1)  # normalize to 0-1 values

    # Compute edge
    Edge_img = sobel(img)
    Edge_img = normalizeRange(Edge_img, 0, 1)

    # Superimpose intensity and edge images
    "combine the intensity image and edge image into a single image that preserves the important information from both images. "
    combined_img = w_line * img + w_edge * Edge_img
    """interpolate the combined image in order to make it smoother. 
    The RectBivariateSpline function is used to create a 2D interpolation function, 
    which takes the combined image as input and generates a smooth interpolated function. 
    The kx and ky parameters specify the degree of smoothing to be applied along each axis,
     while the s parameter controls the smoothness of the output."""
    # Interpolate for smoothness
    interpolate = RectBivariateSpline(
        np.arange(combined_img.shape[1]),
        np.arange(combined_img.shape[0]),
        combined_img.T, kx=2, ky=2, s=0
    )

    # Get snake contour axes
    x1, y1 = inital_contour[:, 0].astype(np.float), inital_contour[:, 1].astype(np.float)

    # store snake progress
    snake_contour = np.array([x1, y1]).T
    contour_history['snakes'] = []
    contour_history['snakes'].append(
        snake_contour)  # keep track of the snake's progress during the optimization process.

    # Build finite difference matrices
    """The matrices A and B represent the second and fourth order derivative operators respectively,
     using the central difference scheme. 
     The parameter alpha and beta control the weighting of the second and fourth order derivatives. Finally, 
     the matrix Z is the combination of A and B matrices with appropriate weights."""
    second_derivative_matrix = np.roll(np.eye(len(x1)), -1, axis=0) + np.roll(np.eye(len(x1)), -1, axis=1) - 2 * np.eye(
        len(x1))  # second order derivative, central difference
    fourth_derivative_matrix = np.roll(np.eye(len(x1)), -2, axis=0) + np.roll(np.eye(len(x1)), -2,
                                                                              axis=1) - 4 * np.roll(np.eye(len(x1)), -1,
                                                                                                    axis=0) - \
                               4 * np.roll(np.eye(len(x1)), -1, axis=1) + \
                               6 * np.eye(len(x1))
    energy_matrix = -alpha * second_derivative_matrix + beta * fourth_derivative_matrix

    # Calculate inverse
    energy_matrix_inv = scipy.linalg.inv(np.eye(len(x1)) + tau * energy_matrix)

    # Snake energy minimization
    x = np.copy(x1)
    y = np.copy(y1)

    """The energy minimization is performed iteratively for a maximum number of iterations specified by max_iter. In each iteration, 
    the function interpolates the energy map using RectBivariateSpline() and computes the external force acting on the snake contour. 
    It then updates the position of each point in the contour using the inverse of the finite difference matrices and the computed external force.
    The movements of each point are capped to a maximum distance specified by max_px_move. The function stores the position of the snake contour after each iteration in the self.hist dictionary.
    Finally, the function returns the final position of the snake contour as an n x 2 array."""
    i = 0
    while True:
        force_x = interpolate(x, y, dx=1, grid=False)
        force_y = interpolate(x, y, dy=1, grid=False)
        force_x = force_x / (np.linalg.norm(force_x) + 1e-6)
        force_y = force_y / (np.linalg.norm(force_y) + 1e-6)

        new_x = np.dot(energy_matrix_inv, x + tau * force_x)
        new_y = np.dot(energy_matrix_inv, y + tau * force_y)

        move_x = max_px_move * np.tanh(new_x - x)
        move_y = max_px_move * np.tanh(new_y - y)

        x += move_x
        y += move_y

        snake_contour = np.array([x, y]).T
        contour_history['snakes'].append(snake_contour)

        i += 1
        if i == max_iter:
            break
    snake = np.array([x, y]).T

    return snake
######################################

def visualization_cotours(Img, cotours, c=(0, 0, 255)):
    R = Img.copy()

    for s in cotours:
        [v1, v2] = s
        v1 = int(v1)
        v2 = int(v2)

        cv2.circle(R, (v1, v2), 3, c)

    return R


def active_contour(img, initialized_contour):
    Img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Run Snakes
    # sn = Snakes()
    cotours = sneak(
        Img, initialized_contour,
        alpha=0.1, beta=3, tau=100,
        w_line=0.0, w_edge=1.0, max_iter=1000
    )

    R = visualization_cotours(img, cotours, c=(0, 0, 255))

    return R


fig, ax = plt.subplots()
circ = None
x1, y1 = None, None
num_points = 400  # number of points on the circle
circle_points = np.zeros((num_points, 2), dtype=np.float32)

img = plt.imread('test 1.png')



cont=[]
center=(0,0)
radius=0
def onclick(event):
    global circ, x1, y1
    if len(cont)==2:
        return 0
    elif event.button == 1 and len(cont)!=2:
        x1, y1 = event.xdata, event.ydata
        cont.append(1)
        print(f'Clicked at x1={x1:.2f}, y1={y1:.2f}')
    elif event.button == 3 and x1 is not None and y1 is not None and len(cont)!=2:
        x2, y2 = event.xdata, event.ydata
        print(f'Clicked at x2={x2:.2f}, y2={y2:.2f}')
        cont.append(1)
        if circ is not None:
            circ.remove()
        radius = ((x2-x1)**2 + (y2-y1)**2)**0.5
        for i in range(num_points):
            theta = 2 * np.pi * i / num_points
            x = x1+ radius * np.cos(theta)
            y = y1 + radius * np.sin(theta)
            circle_points[i, 0] = x
            circle_points[i, 1] = y


        circ = Circle((x1, y1), radius, fill=False, color='b')

        ax.add_patch(circ)
        plt.draw()
        final_snake1 = active_contour(img, circle_points)
        plt.imshow(cv2.cvtColor(final_snake1, cv2.COLOR_BGR2RGB))

        return
ax.imshow(img)
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
