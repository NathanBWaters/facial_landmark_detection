'''
Helps visualize the data being fed to the models
'''
import streamlit as st
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

from landmarks.constants import INPUT_DIMS
from landmarks.model import LandmarkModel
from landmarks.loader import get_datasets

torch.set_printoptions(precision=3, sci_mode=False)


def fig2data(fig):
    '''
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels
           and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    '''
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    # import pdb; pdb.set_trace()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to
    # have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    '''
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    '''
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGB", (w, h), buf.tobytes())


def render_face(title, image, label):
    '''
    Renders the image, its labels and the predictions
    '''
    plt.clf()
    image = image.reshape((INPUT_DIMS[0], INPUT_DIMS[1]))
    imgplot = plt.imshow(image, cmap='gray')
    plt.title(title)
    label_xs = label[0:][::2]
    label_ys = label[1:][::2]
    plt.scatter(x=label_xs, y=label_ys, c='r', s=40)
    return fig2img(imgplot.figure)


def render_data_loader():
    '''
    Renders synthetic letters
    '''
    with torch.no_grad():
        training, validation, test = get_datasets()
        model = LandmarkModel(load_weights=False)
        model.eval()
        for i in range(15):
            feature, label = training[i]
            output = model(torch.unsqueeze(feature, 0))
            image = feature.numpy().transpose(1, 2, 0)
            st.image(image, width=128, clamp=True)
            st.image(render_face('label', image, label),
                     width=512,
                     clamp=True)
            st.image(render_face('prediction', image, output[0]),
                     width=512,
                     clamp=True)
            st.write('Label: {}'.format(str(label)))
            st.write('Output: {}'.format(str(output)))

            st.write('-' * 80)


if __name__ == '__main__':
    render_data_loader()
