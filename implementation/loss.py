import tensorflow as tf

def smooth_l1(x):

    def func1():
        return x**2 * 0.5

    def func2():
        return tf.abs(x) - tf.constant(0.5)

    def f(x): return tf.cond(tf.less(tf.abs(x), tf.constant(1.0)), func1, func2)

    return tf.map_fn(f, x)

def smooth_l1_loss(true, pred):

    """
    compute smooth_l1 loss 
    g : ground truth
    p = prediction

    hat{g}(x) = (g_x - p_x) / p_w
    hat{g}(y) = (g_y - p_y) / p_hi
    hat{g}(w) = log(g_w / p_w)
    hat{g}(h) = log(g_h / p_h)

    smooth_l1_loss = reduce_mean(smooth_l1(g - hat{g}))
    """

    face_label = true[:, :, :1]

    gxs = true[:, :, 1:2]
    gys = true[:, :, 2:3]
    gws = true[:, :, 3:4]
    ghs = true[:, :, 4:5]

    pxs = pred[:, :, 1:2]
    pys = pred[:, :, 2:3]
    pws = pred[:, :, 3:4]
    phs = pred[:, :, 4:5]

    logitx = (gxs - pxs) / pws
    logity = (gys - pys) / phs
    logitw = tf.math.log(gws / pws)
    logith = tf.math.log(ghs / phs)

    lossx = face_label * \
        tf.map_fn(smooth_l1, tf.reshape(gxs - logitx, (-1, 896)))
    lossy = face_label * \
        tf.map_fn(smooth_l1, tf.reshape(gys - logity, (-1, 896)))
    lossw = face_label * \
        tf.map_fn(smooth_l1, tf.reshape(gws - logitw, (-1, 896)))
    lossh = face_label * \
        tf.map_fn(smooth_l1, tf.reshape(ghs - logith, (-1, 896)))

    x_sum = tf.reduce_sum(lossx)
    y_sum = tf.reduce_sum(lossy)
    w_sum = tf.reduce_sum(lossw)
    h_sum = tf.reduce_sum(lossh)

    loss = tf.stack((x_sum, y_sum, w_sum, h_sum))

    return tf.reduce_mean(loss)


if __name__ == "__main__":
    pass

