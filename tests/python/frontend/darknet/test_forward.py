# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Test Darknet Models
===================
This article is a test script to test darknet models with Relay.
All the required models and libraries will be downloaded from the internet
by the script.
"""
import numpy as np
import tvm
from tvm import te
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata
download_testdata.__test__ = False
from tvm.relay.testing.darknet import LAYERTYPE
from tvm.relay.testing.darknet import __darknetffi__
from tvm.relay.frontend.darknet import ACTIVATION
from tvm import relay

REPO_URL = 'https://github.com/siju-samuel/darknet/blob/tvm_yolo/'
DARKNET_LIB = 'libdarknet3.0.so'
DARKNETLIB_URL = REPO_URL + 'lib/' + DARKNET_LIB + '?raw=true'
LIB = __darknetffi__.dlopen(download_testdata(DARKNETLIB_URL, DARKNET_LIB, module='darknet'))

DARKNET_TEST_IMAGE_NAME = 'dog.jpg'
DARKNET_TEST_IMAGE_URL = REPO_URL + 'data/' + DARKNET_TEST_IMAGE_NAME +'?raw=true'
DARKNET_TEST_IMAGE_PATH = download_testdata(DARKNET_TEST_IMAGE_URL, DARKNET_TEST_IMAGE_NAME, module='data')

def _read_memory_buffer(shape, data, dtype='float32'):
    length = 1
    for x in shape:
        length *= x
    data_np = np.zeros(length, dtype=dtype)
    for i in range(length):
        data_np[i] = data[i]
    return data_np.reshape(shape)


def _get_tvm_output(net, data, build_dtype='float32', states=None):
    '''Compute TVM output'''
    dtype = 'float32'
    mod, params = relay.frontend.from_darknet(net, data.shape, dtype)
    target = 'llvm'
    shape_dict = {'data': data.shape}
    graph, library, params = relay.build(mod,
                                         target,
                                         params=params)

    # Execute on TVM
    ctx = tvm.cpu(0)
    m = graph_runtime.create(graph, library, ctx)
    # set inputs
    m.set_input('data', tvm.nd.array(data.astype(dtype)))
    if states:
        for name in states.keys():
            m.set_input(name, tvm.nd.array(states[name].astype(dtype)))
    m.set_input(**params)
    m.run()
    # get outputs
    tvm_out = []
    for i in range(m.get_num_outputs()):
        tvm_out.append(m.get_output(i).asnumpy())
    return tvm_out


def _load_net(cfg_url, cfg_name, weights_url, weights_name):
    cfg_path = download_testdata(cfg_url, cfg_name, module='darknet')
    weights_path = download_testdata(weights_url, weights_name, module='darknet')
    net = LIB.load_network_custom(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0, 1)
    return net


def verify_darknet_frontend(net, build_dtype='float32'):
    '''Test network with given input image on both darknet and tvm'''
    def get_darknet_output(net, img):
        LIB.network_predict_image(net, img)
        out = []
        for i in range(net.n):
            layer = net.layers[i]
            if layer.type == LAYERTYPE.REGION:
                attributes = np.array([layer.n, layer.out_c, layer.out_h,
                                       layer.out_w, layer.classes,
                                       layer.coords, layer.background],
                                      dtype=np.int32)
                out.insert(0, attributes)
                out.insert(0, _read_memory_buffer((layer.n*2, ), layer.biases))
                layer_outshape = (layer.batch, layer.out_c,
                                  layer.out_h, layer.out_w)
                out.insert(0, _read_memory_buffer(layer_outshape, layer.output))
            elif layer.type == LAYERTYPE.YOLO:
                attributes = np.array([layer.n, layer.out_c, layer.out_h,
                                       layer.out_w, layer.classes,
                                       layer.total],
                                      dtype=np.int32)
                out.insert(0, attributes)
                out.insert(0, _read_memory_buffer((layer.total*2, ), layer.biases))
                out.insert(0, _read_memory_buffer((layer.n, ), layer.mask, dtype='int32'))
                layer_outshape = (layer.batch, layer.out_c,
                                  layer.out_h, layer.out_w)
                out.insert(0, _read_memory_buffer(layer_outshape, layer.output))
            elif i == net.n-1:
                if layer.type == LAYERTYPE.CONNECTED:
                    darknet_outshape = (layer.batch, layer.out_c)
                elif layer.type in [LAYERTYPE.SOFTMAX]:
                    darknet_outshape = (layer.batch, layer.outputs)
                else:
                    darknet_outshape = (layer.batch, layer.out_c,
                                        layer.out_h, layer.out_w)
                out.insert(0, _read_memory_buffer(darknet_outshape, layer.output))
        return out

    dtype = 'float32'

    img = LIB.letterbox_image(LIB.load_image_color(DARKNET_TEST_IMAGE_PATH.encode('utf-8'), 0, 0), net.w, net.h)
    darknet_outs = get_darknet_output(net, img)
    batch_size = 1
    data = np.empty([batch_size, img.c, img.h, img.w], dtype)
    i = 0
    for c in range(img.c):
        for h in range(img.h):
            for k in range(img.w):
                data[0][c][h][k] = img.data[i]
                i = i + 1

    tvm_outs = _get_tvm_output(net, data, build_dtype)
    for tvm_out, darknet_out in (zip(tvm_outs, darknet_outs)):
        tvm.testing.assert_allclose(darknet_out, tvm_out, rtol=1e-3, atol=1e-3)


def test_forward_convolutional():
    '''test convolutional layer'''
    net = LIB.make_network_custom(1)
    batch = 1
    steps = 1
    h = 224
    w = 224
    c = 3
    n = 32
    groups = 1
    size = 3
    stride_x =2
    stride_y = 2
    dilation = 1
    padding  = 0
    activation = 1
    batch_normalize  = 0
    binary = 0
    xnor = 0
    adam = 0
    use_bin_output = 0
    index = 0
    antialiasing = 0
    share_layer = __darknetffi__.NULL
    assisted_excitation = 0
    deform = 0
    train = 0

    layer = LIB.make_convolutional_layer(batch, steps, h, w, c, n, groups, size, stride_x, stride_y,
                                         dilation, padding, activation, batch_normalize, binary,
                                         xnor, adam, use_bin_output, index, antialiasing, share_layer,
                                         assisted_excitation, deform, train)
    net.layers[0] = layer
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    verify_darknet_frontend(net)
    LIB.free_network(net)


def test_forward_dense():
    '''test fully connected layer'''
    net = LIB.make_network_custom(1)

    batch = 1
    steps = 1
    inputs = 48
    outputs = 16
    activation = 1
    batch_normalize = 0

    layer = LIB.make_connected_layer(batch, steps, inputs, outputs, activation, batch_normalize)
    net.layers[0] = layer
    net.c = 3
    net.w = net.h = 4
    verify_darknet_frontend(net)
    LIB.free_network(net)


def test_forward_dense_batchnorm():
    '''test fully connected layer with batchnorm'''
    net = LIB.make_network_custom(1)
    batch = 1
    steps = 1
    inputs = 12
    outputs = 2
    activation = 1
    batch_normalize = 1
    layer = LIB.make_connected_layer(batch, steps, inputs, outputs, activation, batch_normalize)
    for i in range(5):
        layer.rolling_mean[i] = np.random.rand(1)
        layer.rolling_variance[i] = np.random.rand(1) + 0.5
        layer.scales[i] = np.random.rand(1)
    net.layers[0] = layer
    net.w = net.h = 2
    verify_darknet_frontend(net)
    LIB.free_network(net)


def test_forward_maxpooling():
    '''test maxpooling layer'''
    net = LIB.make_network_custom(1)

    batch = 1
    h = 224
    w = 224
    c = 3
    size = 2
    stride_x = 2
    stride_y = 2
    padding = 0
    maxpool_depth = 0
    out_channels = 0
    antialiasing = 0
    avgpool = 0
    train = 0

    layer = LIB.make_maxpool_layer(batch, h, w, c, size, stride_x, stride_y, padding, maxpool_depth, out_channels, antialiasing, avgpool, train)
    net.layers[0] = layer
    net.w = w
    net.h = h
    LIB.resize_network(net, h, w)
    verify_darknet_frontend(net)
    LIB.free_network(net)


def test_forward_avgpooling():
    '''test avgerage pooling layer'''
    net = LIB.make_network_custom(1)
    batch = 1
    h = 224
    w = 224
    c = 3
    layer = LIB.make_avgpool_layer(batch, h, w, c)
    net.layers[0] = layer
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_conv_batch_norm():
    '''test batch normalization layer'''
    net = LIB.make_network_custom(1)
    batch = 1
    steps = 1
    h = 224
    w = 224
    c = 3
    n = 32
    groups = 1
    size = 3
    stride_x =2
    stride_y = 2
    dilation = 1
    padding  = 0
    activation = 1
    batch_normalize  = 1
    binary = 0
    xnor = 0
    adam = 0
    use_bin_output = 0
    index = 0
    antialiasing = 0
    share_layer = __darknetffi__.NULL
    assisted_excitation = 0
    deform = 0
    train = 0

    layer = LIB.make_convolutional_layer(batch, steps, h, w, c, n, groups, size, stride_x, stride_y,
                                         dilation, padding, activation, batch_normalize, binary,
                                         xnor, adam, use_bin_output, index, antialiasing, share_layer,
                                         assisted_excitation, deform, train)


    for i in range(32):
        layer.rolling_mean[i] = np.random.rand(1)
        layer.rolling_variance[i] = np.random.rand(1) + 0.5
    net.layers[0] = layer
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    verify_darknet_frontend(net)
    LIB.free_network(net)


def test_forward_yolo_op():
    '''test yolo layer'''
    net = LIB.make_network_custom(2)
    batch = 1
    steps = 1
    h = 224
    w = 224
    c = 3
    n = 14
    groups = 1
    size = 3
    stride_x = 2
    stride_y = 2
    dilation = 1
    padding  = 0
    activation = 1
    batch_normalize  = 0
    binary = 0
    xnor = 0
    adam = 0
    use_bin_output = 0
    index = 0
    antialiasing = 0
    share_layer = __darknetffi__.NULL
    assisted_excitation = 0
    deform = 0
    train = 0

    layer_1 = LIB.make_convolutional_layer(batch, steps, h, w, c, n, groups, size, stride_x, stride_y,
                                           dilation, padding, activation, batch_normalize, binary,
                                           xnor, adam, use_bin_output, index, antialiasing, share_layer,
                                           assisted_excitation, deform, train)

    layer_2 = LIB.make_yolo_layer(1, 111, 111, 2, 9, __darknetffi__.NULL, 2, 0)
    net.layers[0] = layer_1
    net.layers[1] = layer_2
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    build_dtype = {}
    verify_darknet_frontend(net, build_dtype)
    LIB.free_network(net)


def test_forward_upsample():
    '''test upsample layer'''
    net = LIB.make_network_custom(1)
    layer = LIB.make_upsample_layer(1, 19, 19, 3, 3)
    layer.scale = 1
    net.layers[0] = layer
    net.w = net.h = 19
    LIB.resize_network(net, 19, 19)
    verify_darknet_frontend(net)
    LIB.free_network(net)


def test_forward_elu():
    '''test elu activation layer'''
    net = LIB.make_network_custom(1)
    batch = 1
    steps = 1
    h = 224
    w = 224
    c = 3
    n = 32
    groups = 1
    size = 3
    stride_x =2
    stride_y = 2
    dilation = 1
    padding  = 0
    activation = 1
    batch_normalize  = 0
    binary = 0
    xnor = 0
    adam = 0
    use_bin_output = 0
    index = 0
    antialiasing = 0
    share_layer = __darknetffi__.NULL
    assisted_excitation = 0
    deform = 0
    train = 0

    layer_1 = LIB.make_convolutional_layer(batch, steps, h, w, c, n, groups, size, stride_x, stride_y,
                                           dilation, padding, activation, batch_normalize, binary,
                                           xnor, adam, use_bin_output, index, antialiasing, share_layer,
                                           assisted_excitation, deform, train)
    layer_1.activation = ACTIVATION.ELU
    net.layers[0] = layer_1
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    verify_darknet_frontend(net)
    LIB.free_network(net)


def test_forward_softmax():
    '''test softmax layer'''
    net = LIB.make_network_custom(1)
    layer_1 = LIB.make_softmax_layer(1, 75, 1)
    layer_1.temperature = 1
    net.layers[0] = layer_1
    net.w = net.h = 5
    verify_darknet_frontend(net)
    LIB.free_network(net)


def test_forward_softmax_temperature():
    '''test softmax layer'''
    net = LIB.make_network_custom(1)
    layer_1 = LIB.make_softmax_layer(1, 75, 1)
    layer_1.temperature = 0.8
    net.layers[0] = layer_1
    net.w = net.h = 5
    verify_darknet_frontend(net)
    LIB.free_network(net)


def test_forward_activation_logistic():
    '''test logistic activation layer'''
    net = LIB.make_network_custom(1)
    batch = 1
    steps = 1
    h = 224
    w = 224
    c = 3
    n = 32
    groups = 1
    size = 3
    stride_x =2
    stride_y = 2
    dilation = 1
    padding  = 0
    activation = ACTIVATION.LOGISTIC
    batch_normalize  = 0
    binary = 0
    xnor = 0
    adam = 0
    use_bin_output = 0
    index = 0
    antialiasing = 0
    share_layer = __darknetffi__.NULL
    assisted_excitation = 0
    deform = 0
    train = 0
    layer_1 = LIB.make_convolutional_layer(batch, steps, h, w, c, n, groups, size, stride_x, stride_y,
                                           dilation, padding, activation, batch_normalize, binary,
                                           xnor, adam, use_bin_output, index, antialiasing, share_layer,
                                           assisted_excitation, deform, train)

    net.layers[0] = layer_1
    net.w = w
    net.h = h
    LIB.resize_network(net, net.w, net.h)
    verify_darknet_frontend(net)
    LIB.free_network(net)


def test_forward_extraction():
    '''test extraction model'''
    model_name = 'extraction'
    cfg_name = model_name + '.cfg'
    weights_name = model_name + '.weights'
    cfg_url = 'https://github.com/pjreddie/darknet/blob/master/cfg/' + cfg_name + '?raw=true'
    weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'
    net = _load_net(cfg_url, cfg_name, weights_url, weights_name)
    verify_darknet_frontend(net)
    LIB.free_network(net)


def test_forward_alexnet():
    '''test alexnet model'''
    model_name = 'alexnet'
    cfg_name = model_name + '.cfg'
    weights_name = model_name + '.weights'
    cfg_url = 'https://github.com/pjreddie/darknet/blob/master/cfg/' + cfg_name + '?raw=true'
    weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'
    net = _load_net(cfg_url, cfg_name, weights_url, weights_name)
    verify_darknet_frontend(net)
    LIB.free_network(net)


def test_forward_yolov3():
    '''test yolov3 model'''
    model_name = 'yolov3'
    cfg_name = model_name + '.cfg'
    weights_name = model_name + '.weights'
    cfg_url = 'https://github.com/pjreddie/darknet/blob/master/cfg/' + cfg_name + '?raw=true'
    weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'
    net = _load_net(cfg_url, cfg_name, weights_url, weights_name)
    build_dtype = {}
    verify_darknet_frontend(net, build_dtype)
    LIB.free_network(net)


if __name__ == '__main__':
    test_forward_convolutional()
    test_forward_maxpooling()
    test_forward_avgpooling()
    test_forward_conv_batch_norm()
    test_forward_dense()
    test_forward_dense_batchnorm()
    test_forward_softmax()
    test_forward_softmax_temperature()
    test_forward_yolo_op()
    test_forward_upsample()
    test_forward_elu()
    test_forward_activation_logistic()

    # pretrained models
    test_forward_alexnet()
    test_forward_extraction()
    test_forward_yolov3()
