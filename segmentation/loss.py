import tensorflow as tf


def sparse_ce_logits_with_ignore(y_true, logits, ignore=0):
    raw_prediction = tf.reshape(logits, [-1, tf.shape(logits)[-1]])
    gt = tf.reshape(y_true, [-1])

    indices = tf.squeeze(tf.where(tf.not_equal(gt, ignore)), 1)
    gt = tf.cast(tf.gather(gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)
    loss = tf.reduce_mean(
        (tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt, name="entropy")))
    return loss
