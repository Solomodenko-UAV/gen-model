

def calc_conv_layer_out_dim(image_dim: tuple, padding: int, kernel_size: int, stride: int):
    image_width, image_height = image_dim
    out_width = int((image_width + 2 * padding - kernel_size) / stride + 1)
    out_height = int((image_height + 2 * padding - kernel_size) / stride + 1)
    return out_width, out_height