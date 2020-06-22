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
# pylint: disable=invalid-name, unused-variable, unused-argument, no-init
"""
Compile DarkNet Models
====================
DarkNet helper functions for darknet model parsing and image loading.
This functions will not be loaded by default.
These are utility functions used for testing and tutorial file.
"""
from __future__ import division
import numpy as np
from cffi import FFI
import cv2


def convert_image(image):
    """Convert the image with numpy."""
    imagex = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imagex = np.array(image)
    imagex = imagex.transpose((2, 0, 1))
    imagex = np.divide(imagex, 255.0)
    imagex = np.flip(imagex, 0)
    return imagex

def load_image_color(test_image):
    """To load the image using opencv api and do preprocessing."""
    imagex = cv2.imread(test_image)
    return convert_image(imagex)

def _letterbox_image(img, w_in, h_in):
    """To get the image in boxed format."""
    imh, imw, imc = img.shape
    if (w_in / imw) < (h_in / imh):
        new_w = w_in
        new_h = imh * w_in // imw
    else:
        new_h = h_in
        new_w = imw * h_in // imh
    dim = (new_w, new_h)
    # Default interpolation method is INTER_LINEAR
    # Other methods are INTER_AREA, INTER_NEAREST, INTER_CUBIC and INTER_LANCZOS4
    # For more information see:
    # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize
    resized = cv2.resize(src=img, dsize=dim, interpolation=cv2.INTER_CUBIC)
    resized = convert_image(resized)
    boxed = np.full((imc, h_in, w_in), 0.5, dtype=float)
    _, resizedh, resizedw = resized.shape
    boxed[:, int((h_in - new_h) / 2)
          :int((h_in - new_h) / 2) + resizedh, int((w_in - new_w) / 2)
          :int((w_in - new_w) / 2) + resizedw] = resized
    return boxed

def load_image(img, resize_width, resize_height):
    """Load the image and convert to the darknet model format.
    The image processing of darknet is different from normal.
    Parameters
    ----------
    image : string
        The image file name with path

    resize_width : integer
        The width to which the image needs to be resized

    resize_height : integer
        The height to which the image needs to be resized

    Returns
    -------
    img : Float array
        Array of processed image
    """
    imagex = cv2.imread(img)
    return _letterbox_image(imagex, resize_width, resize_height)

class LAYERTYPE(object):
    """Darknet LAYERTYPE Class constant."""
    CONVOLUTIONAL = 0
    DECONVOLUTIONAL = 1
    CONNECTED = 2
    MAXPOOL = 3
    LOCAL_AVGPOOL = 4
    SOFTMAX = 5
    DETECTION = 6
    DROPOUT = 7
    CROP = 8
    ROUTE = 9
    COST = 10
    NORMALIZATION = 11
    AVGPOOL = 12
    LOCAL = 13
    SHORTCUT = 14
    SCALE_CHANNELS = 15
    SAM = 16
    ACTIVE = 17
    RNN = 18
    GRU = 19
    LSTM = 20
    CONV_LSTM = 21
    CRNN = 22
    BATCHNORM = 23
    NETWORK = 24
    XNOR = 25
    REGION = 26
    YOLO = 27
    GAUSSIAN_YOLO = 28
    ISEG = 29
    REORG = 30
    REORG_OLD = 31
    UPSAMPLE = 32
    LOGXENT = 33
    L2NORM = 34
    EMPTY = 35
    BLANK = 36

class ACTIVATION(object):
    """Darknet ACTIVATION Class constant."""
    LOGISTIC = 0
    RELU = 1
    RELU6 = 2
    RELIE = 3
    LINEAR = 4
    RAMP = 5
    TANH = 6
    PLSE = 7
    LEAKY = 8
    ELU = 9
    LOGGY = 10
    STAIR = 11
    HARDTAN = 12
    LHTAN = 13
    SELU = 14
    GELU = 15
    SWISH = 16
    MISH = 17
    NORM_CHAN = 18
    NORM_CHAN_SOFTMAX = 19
    NORM_CHAN_SOFTMAX_MAXVAL = 20


__darknetffi__ = FFI()

__darknetffi__.cdef("""
typedef struct network network;
typedef struct layer layer;

typedef struct{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
} ACTIVATION;


typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    LOCAL_AVGPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    SCALE_CHANNELS,
    SAM,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CONV_LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    GAUSSIAN_YOLO,
    ISEG,
    REORG,
    REORG_OLD,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    EMPTY,
    BLANK
} LAYERTYPE;

typedef enum{
    SSE, MASKED, L1, SEG, SMOOTH, WGAN
} COSTTYPE;


struct layer{
    LAYERTYPE type;
    ACTIVATION activation;
    ACTIVATION lstm_activation;
    COSTTYPE cost_type;
    void (*forward);
    void (*backward);
    void (*update);
    void (*forward_gpu);
    void (*backward_gpu);
    void (*update_gpu);
    layer *share_layer;
    int train;
    int avgpool;
    int batch_normalize;
    int shortcut;
    int batch;
    int dynamic_minibatch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    float mean_alpha;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h, w, c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int group_id;
    int size;
    int side;
    int stride;
    int stride_x;
    int stride_y;
    int dilation;
    int antialiasing;
    int maxpool_depth;
    int out_channels;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int scale_wh;
    int binary;
    int xnor;
    int peephole;
    int use_bin_output;
    int keep_delta_gpu;
    int optimized_memory;
    int steps;
    int bottleneck;
    float time_normalizer;
    int state_constrain;
    int hidden;
    int truth;
    float smooth;
    float dot;
    int deform;
    int grad_centr;
    int sway;
    int rotate;
    int stretch;
    int stretch_sway;
    float angle;
    float jitter;
    float resize;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int focal_loss;
    float *classes_multipliers;
    float label_smooth_eps;
    int noloss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int does_cost;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;
    float bflops;

    int adam;
    float B1;
    float B2;
    float eps;

    int t;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    float random;
    float ignore_thresh;
    float truth_thresh;
    float iou_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;
    int assisted_excitation;

    int onlyforward;
    int stopbackward;
    int train_only_bn;
    int dont_update;
    int burnin_update;
    int dontload;
    int dontsave;
    int dontloadscales;
    int numload;

    float temperature;
    float probability;
    float dropblock_size_rel;
    int dropblock_size_abs;
    int dropblock;
    float scale;

    int receptive_w;
    int receptive_h;
    int receptive_w_scale;
    int receptive_h_scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    float **layers_output;
    float **layers_delta;
    int weights_type;
    int weights_normalization;
    int   * map;
    int   * counts;
    float ** sums;
    float * rand;
    float * cost;
    int *labels;
    float *cos_sim;
    float *p_constrastive;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float *concat;
    float *concat_delta;

    float *binary_weights;

    float *biases;
    float *bias_updates;

    float *scales;
    float *scale_updates;

    float *weights;
    float *weight_updates;

    float scale_x_y;
    int objectness_smooth;
    float max_delta;
    float uc_normalizer;
    float iou_normalizer;
    float cls_normalizer;
    int iou_loss;
    int iou_thresh_kind;
    int nms_kind;
    float beta_nms;
    int yolo_point;

    char *align_bit_weights_gpu;
    float *mean_arr_gpu;
    float *align_workspace_gpu;
    float *transposed_align_workspace_gpu;
    int align_workspace_size;

    char *align_bit_weights;
    float *mean_arr;
    int align_bit_weights_size;
    int lda_align;
    int new_lda;
    int bit_align;

    float *col_image;
    float * delta;
    float * output;
    float * activation_input;
    int delta_pinned;
    int output_pinned;
    float * loss;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    float * m;
    float * v;

    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;


    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float *stored_h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *stored_c_cpu;
    float *dc_cpu;

    float *binary_input;
    uint32_t *bin_re_packed_input;
    char *t_bit_input;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;

    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *vo;
    struct layer *uf;
    struct layer *wf;
    struct layer *vf;
    struct layer *ui;
    struct layer *wi;
    struct layer *vi;
    struct layer *ug;
    struct layer *wg;

    tree *softmax_tree;

    size_t workspace_size;

};


typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} LEARNINGRATEPOLICY;

typedef struct network{
    int n;
    int batch;
    uint64_t *seen;
    int *cur_iteration;
    float loss_scale;
    int *t;
    float epoch;
    int subdivisions;
    layer *layers;
    float *output;
    LEARNINGRATEPOLICY policy;
    int benchmark_layers;
    int *total_bbox;
    int *rewritten_bbox;

    float learning_rate;
    float learning_rate_min;
    float learning_rate_max;
    int batches_per_cycle;
    int batches_cycle_mult;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    int num_boxes;
    int train_images_num;
    float *seq_scales;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;
    int cudnn_half;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    int flip; // horizontal flip 50% probability augmentaiont for classifier training (default = 1)
    int gaussian_noise;
    int blur;
    int mixup;
    float label_smooth_eps;
    int resize_step;
    int attention;
    int adversarial;
    float adversarial_lr;
    float max_chart_loss;
    int letter_box;
    int mosaic_bound;
    int contrastive;
    int unsupervised;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;
    int track;
    int augment_speed;
    int sequential_subdivisions;
    int init_sequential_subdivisions;
    int current_subdivision;
    int try_fix_nan;

    int gpu_index;
    tree *hierarchy;

    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;
    float clip;
} network;


typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

network *load_network(char *cfg, char *weights, int clear);
network *load_network_custom(char *cfg, char *weights, int clear, int batch);
image letterbox_image(image im, int w, int h);
int resize_network(network *net, int w, int h);
void top_predictions(network *net, int n, int *index);
void free_image(image m);
image load_image_color(char *filename, int w, int h);
float *network_predict_image(network *net, image im);
float *network_predict(network *net, float *input);
network *make_network_custom(int n);
layer make_convolutional_layer(int batch, int steps, int h, int w, int c, int n, int groups, int size, int stride_x, int stride_y, int dilation, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index, int antialiasing, int *share_layer, int assisted_excitation, int deform, int train);
layer make_connected_layer(int batch, int steps, int inputs, int outputs, ACTIVATION activation, int batch_normalize);
layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride_x, int stride_y, int padding, int maxpool_depth, int out_channels, int antialiasing, int avgpool, int train);
layer make_avgpool_layer(int batch, int w, int h, int c);
layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
layer make_batchnorm_layer(int batch, int w, int h, int c);
layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse);
layer make_region_layer(int batch, int w, int h, int n, int classes, int coords, int max_boxes);
layer make_softmax_layer(int batch, int inputs, int groups);
layer make_rnn_layer(int batch, int inputs, int hidden, int outputs, int steps, ACTIVATION activation, int batch_normalize, int log);
layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes);
layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int steps, ACTIVATION activation, int batch_normalize);
layer make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam);
layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam);
layer make_upsample_layer(int batch, int w, int h, int c, int stride);
layer make_l2norm_layer(int batch, int inputs);
void free_network(network *net);
"""
                   )
