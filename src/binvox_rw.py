"""Read .binvox files into numpy arrays.

Adapted from the public-domain binvox-rw-py by Daniel Maturana.
https://github.com/dimatura/binvox-rw-py
"""

import numpy as np


class Voxels:
    def __init__(self, data, dims, translate, scale, axis_order):
        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale
        self.axis_order = axis_order


def read_as_3d_array(fp):
    header = fp.readline().strip()
    if hasattr(header, 'decode'):
        header = header.decode('utf-8')
    if not header.startswith('#binvox'):
        raise IOError('Not a binvox file')

    dims_line = fp.readline().strip()
    if hasattr(dims_line, 'decode'):
        dims_line = dims_line.decode('utf-8')
    dims = [int(x) for x in dims_line.split()[1:]]

    translate_line = fp.readline().strip()
    if hasattr(translate_line, 'decode'):
        translate_line = translate_line.decode('utf-8')
    translate = [float(x) for x in translate_line.split()[1:]]

    scale_line = fp.readline().strip()
    if hasattr(scale_line, 'decode'):
        scale_line = scale_line.decode('utf-8')
    scale = float(scale_line.split()[1])

    data_line = fp.readline().strip()
    if hasattr(data_line, 'decode'):
        data_line = data_line.decode('utf-8')

    raw = np.frombuffer(fp.read(), dtype=np.uint8)
    values = raw[::2].astype(bool)
    counts = raw[1::2].astype(int)

    flat = np.repeat(values, counts).astype(np.float32)
    volume = flat.reshape(dims)

    return Voxels(volume, dims, translate, scale, 'xzy')
