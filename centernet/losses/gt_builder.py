import tensorflow as tf

from centernet.ops import preprocessing_ops


def _build_heatmap_and_regressed_features(labels,
                                          output_size=[128, 128], 
                                          input_size=[512, 512],
                                          num_classes=90,
                                          max_num_instances=128,
                                          use_gaussian_bump=True,
                                          gaussian_rad=-1,
                                          gaussian_iou=0.7,
                                          class_offset=1,
                                          dtype='float32'):
    """ Generates the ground truth labels for centernet.
    
    Ground truth labels are generated by splatting gaussians on heatmaps for
    corners and centers. Regressed features (offsets and sizes) are also
    generated.

    Args:
      labels: A dictionary of COCO ground truth labels with at minimum the following fields:
        bbox: A `Tensor` of shape [max_num_instances, num_boxes, 4], where the last dimension
          corresponds to the top left x, top left y, bottom right x, and
          bottom left y coordinates of the bounding box
        classes: A `Tensor` of shape [max_num_instances, num_boxes] that contains the class of each
          box, given in the same order as the boxes
        num_detections: A `Tensor` or int that gives the number of objects in the image
      output_size: A `list` of length 2 containing the desired output height 
        and width of the heatmaps
      input_size: A `list` of length 2 the expected input height and width of 
        the image
    Returns:
      Dictionary of labels with the following fields:
        'tl_heatmaps': A `Tensor` of shape [output_h, output_w, num_classes],
          heatmap with splatted gaussians centered at the positions and channels
          corresponding to the top left location and class of the object
        'br_heatmaps': `Tensor` of shape [output_h, output_w, num_classes],
          heatmap with splatted gaussians centered at the positions and channels
          corresponding to the bottom right location and class of the object
        'ct_heatmaps': Tensor of shape [output_h, output_w, num_classes],
          heatmap with splatted gaussians centered at the positions and channels
          corresponding to the center location and class of the object
        'tl_offset': `Tensor` of shape [max_num_instances, 2], where the first
          num_boxes entries contain the x-offset and y-offset of the top-left
          corner of an object. All other entires are 0
        'br_offset': `Tensor` of shape [max_num_instances, 2], where the first
          num_boxes entries contain the x-offset and y-offset of the 
          bottom-right corner of an object. All other entires are 0
        'ct_offset': `Tensor` of shape [max_num_instances, 2], where the first
          num_boxes entries contain the x-offset and y-offset of the center of 
          an object. All other entires are 0
        'size': `Tensor` of shape [max_num_instances, 2], where the first
          num_boxes entries contain the width and height of an object. All 
          other entires are 0
        'box_mask': `Tensor` of shape [max_num_instances], where the first
          num_boxes entries are 1. All other entires are 0
        'box_indices': `Tensor` of shape [max_num_instances, 2], where the first
          num_boxes entries contain the y-center and x-center of a valid box. 
          These are used to extract the regressed box features from the 
          prediction when computing the loss
      """
    if dtype == 'float16':
      dtype = tf.float16
    elif dtype == 'bfloat16':
      dtype = tf.bfloat16
    elif dtype == 'float32':
      dtype = tf.float32
    else:
      raise Exception(
        'Unsupported datatype used in ground truth builder only {float16, bfloat16, or float32}'
      )
    # Get relevant bounding box and class information from labels
    # only keep the first num_objects boxes and classes
    num_objects = labels['num_detections']
    boxes = labels['bbox']
    classes = labels['classes'] - class_offset

    # Compute scaling factors for center/corner positions on heatmap
    input_size = tf.cast(input_size, dtype)
    output_size = tf.cast(output_size, dtype)
    input_h, input_w = input_size[0], input_size[1]
    output_h, output_w = output_size[0], output_size[1]

    width_ratio = output_w / input_w
    height_ratio = output_h / input_h
    
    # Original box coordinates
    ytl, ybr = boxes[..., 0], boxes[..., 2]
    xtl, xbr = boxes[..., 1], boxes[..., 3]
    yct = (ytl + ybr) / 2
    xct = (xtl + xbr) / 2

    # Scaled box coordinates (could be floating point)
    fxtl = xtl * width_ratio
    fytl = ytl * height_ratio
    fxbr = xbr * width_ratio
    fybr = ybr * height_ratio
    fxct = xct * width_ratio
    fyct = yct * height_ratio

    # Floor the scaled box coordinates to be placed on heatmaps
    xtl = tf.math.floor(fxtl)
    ytl = tf.math.floor(fytl)
    xbr = tf.math.floor(fxbr)
    ybr = tf.math.floor(fybr)
    xct = tf.math.floor(fxct)
    yct = tf.math.floor(fyct)

    # Offset computations to make up for discretization error
    # used for offset maps
    # tl_offset_values = tf.stack([fxtl - xtl, fytl - ytl], axis=-1)
    # br_offset_values = tf.stack([fxbr - xbr, fybr - ybr], axis=-1)
    ct_offset_values = tf.stack([fxct - xct, fyct - yct], axis=-1)
    
    # Get the scaled box dimensions for computing the gaussian radius
    box_widths = boxes[..., 3] - boxes[..., 1]
    box_heights = boxes[..., 2] - boxes[..., 0]

    box_widths = box_widths * width_ratio
    box_heights = box_heights * height_ratio

    # Used for size map
    box_widths_heights = tf.stack([box_widths, box_heights], axis=-1)
    
    # Center/corner heatmaps 
    # tl_heatmap = tf.zeros((output_h, output_w, num_classes), dtype)
    # br_heatmap = tf.zeros((output_h, output_w, num_classes), dtype)
    ct_heatmap = tf.zeros((output_h, output_w, num_classes), dtype)
    
    # Maps for offset and size features for each instance of a box
    # tl_offset = tf.zeros((max_num_instances, 2), dtype)
    # br_offset = tf.zeros((max_num_instances, 2), dtype)
    ct_offset = tf.zeros((max_num_instances, 2), dtype)
    size = tf.zeros((max_num_instances, 2), dtype)
    
    # Mask for valid box instances and their center indices in the heatmap
    box_mask = tf.zeros((max_num_instances), tf.int32)
    box_indices  = tf.zeros((max_num_instances, 2), tf.int32)

    if use_gaussian_bump:
      # Need to gaussians around the centers and corners of the objects

      # First compute the desired gaussian radius
      if gaussian_rad == -1:
        print(box_widths_heights)
        radius = tf.map_fn(fn=lambda x: preprocessing_ops.gaussian_radius(x), 
          elems=tf.math.ceil(box_widths_heights))
        print(radius)
        radius = tf.math.maximum(tf.math.floor(radius), 
          tf.cast(1.0, radius.dtype))
      else:
        radius = tf.constant([gaussian_rad] * max_num_instances, dtype)
      # These blobs contain information needed to draw the gaussian
      # tl_blobs = tf.stack([classes, xtl, ytl, radius], axis=-1)
      # br_blobs = tf.stack([classes, xbr, ybr, radius], axis=-1)
      ct_blobs = tf.stack([classes, xct, yct, radius], axis=-1)
      
      # Get individual gaussian contributions from each bounding box
      # tl_gaussians = tf.map_fn(
      #   fn=lambda x: preprocessing_ops.draw_gaussian(
      #     tf.shape(tl_heatmap), x, dtype), elems=tl_blobs)
      # br_gaussians = tf.map_fn(
      #   fn=lambda x: preprocessing_ops.draw_gaussian(
      #     tf.shape(br_heatmap), x, dtype), elems=br_blobs)
      ct_gaussians = tf.map_fn(
        fn=lambda x: preprocessing_ops.draw_gaussian(
          tf.shape(ct_heatmap), x, dtype), elems=ct_blobs)

      # Combine contributions into single heatmaps
      # tl_heatmap = tf.math.reduce_max(tl_gaussians, axis=0)
      # br_heatmap = tf.math.reduce_max(br_gaussians, axis=0)
      ct_heatmap = tf.math.reduce_max(ct_gaussians, axis=0)
    
    else:
      # Instead of a gaussian, insert 1s in the center and corner heatmaps
      # tl_hm_update_indices = tf.cast(
      #   tf.stack([ytl, xtl, classes], axis=-1), tf.int32)
      # br_hm_update_indices = tf.cast(
      #   tf.stack([ybr, xbr, classes], axis=-1), tf.int32)
      ct_hm_update_indices = tf.cast(
        tf.stack([yct, xct, classes], axis=-1), tf.int32)

      # tl_heatmap = tf.tensor_scatter_nd_update(tl_heatmap, 
      #   tl_hm_update_indices, [1] * num_objects)
      # br_heatmap = tf.tensor_scatter_nd_update(br_heatmap, 
      #   br_hm_update_indices, [1] * num_objects)
      ct_heatmap = tf.tensor_scatter_nd_update(ct_heatmap, 
        ct_hm_update_indices, [1] * num_objects)
    
    # Indices used to update offsets and sizes for valid box instances 
    update_indices = preprocessing_ops.cartesian_product(
      tf.range(num_objects), tf.range(2))
    update_indices = tf.reshape(update_indices, shape=[num_objects, 2, 2])
    
    # Write the offsets of each box instance
    # tl_offset = tf.tensor_scatter_nd_update(
    #   tl_offset, update_indices, tl_offset_values)
    # br_offset = tf.tensor_scatter_nd_update(
    #   br_offset, update_indices, br_offset_values)
    ct_offset = tf.tensor_scatter_nd_update(
      ct_offset, update_indices, ct_offset_values)

    # Write the size of each bounding box 
    size = tf.tensor_scatter_nd_update(size, update_indices, box_widths_heights)

    # Initially the mask is zeros, so now we unmask each valid box instance
    mask_indices = tf.expand_dims(tf.range(num_objects), -1)
    mask_values = tf.repeat(1, num_objects)
    box_mask = tf.tensor_scatter_nd_update(box_mask, mask_indices, mask_values)

    # Write the y and x coordinate of each box center in the heatmap
    box_index_values = tf.cast(tf.stack([yct, xct], axis=-1), dtype=tf.int32)
    box_indices = tf.tensor_scatter_nd_update(
      box_indices, update_indices, box_index_values)

    labels = {
      # 'tl_heatmaps': tl_heatmap,
      # 'br_heatmaps': br_heatmap,
      'ct_heatmaps': ct_heatmap,
      # 'tl_offset': tl_offset,
      # 'br_offset': br_offset,
      'ct_offset': ct_offset,
      'size': size,
      'box_mask': box_mask,
      'box_indices': box_indices
    }
    return labels
