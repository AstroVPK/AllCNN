import numpy as np
import tensorflow as tf
import datetime
import pdb


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
#(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

height = X_train.shape[1]
width = X_train.shape[2]
channels = X_train.shape[3]
n_inputs = height*width*channels
n_outputs = 10

seed = 42

epsilon = 0.1
S = [200, 250, 300]
n_epochs = 350
batch_size = 100

reg_constant = 0.1

X_train = X_train.astype(np.float32).reshape(-1, height*width*channels) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, height*width*channels) / 255.0
y_train = y_train.astype(np.int32).reshape(-1)
y_test = y_test.astype(np.int32).reshape(-1)
split = X_test.shape[0]//2
X_valid, X_test = X_test[:split], X_test[split:]
y_valid, y_test = y_test[:split], y_test[split:]

# to make this notebook's output stable across runs
def reset_graph(seed=seed):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
#reset_graph()

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
rootLogDir = 'logs'
logDir = '{}/run-{}/'.format(rootLogDir, now)

with tf.name_scope("activation"):
    activation = tf.nn.relu

with tf.name_scope("regularizer"):
    #regularizer = None
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005)
    #regularizer = tf.contrib.layers.l1_regularizer(scale=0.0)

with tf.name_scope("initializer"):
    kernel_init = tf.glorot_uniform_initializer()
    bias_init = tf.zeros_initializer()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")

def AllCNN(X, y, regularizer=regularizer, kernel_init=kernel_init, bias_init=bias_init, activation=activation):

    conv1_fmaps = 96
    conv1_ksize = 3
    conv1_stride = 1
    conv1_pad = "VALID"
    conv1_act = activation

    conv2_fmaps = 96
    conv2_ksize = 3
    conv2_stride = 1
    conv2_pad = "VALID"
    conv2_act = activation

    conv3_fmaps = 96
    conv3_ksize = 3
    conv3_stride = 2
    conv3_pad = "VALID"
    conv3_act = activation

    conv4_fmaps = 192
    conv4_ksize = 3
    conv4_stride = 1
    conv4_pad = "SAME"
    conv4_act = activation

    conv5_fmaps = 192
    conv5_ksize = 3
    conv5_stride = 1
    conv5_pad = "SAME"
    conv5_act = activation

    conv6_fmaps = 192
    conv6_ksize = 3
    conv6_stride = 2
    conv6_pad = "VALID"
    conv6_act = activation

    conv7_fmaps = 192
    conv7_ksize = 3
    conv7_stride = 1
    conv7_pad = "SAME"
    conv7_act = activation

    conv8_fmaps = 192
    conv8_ksize = 1
    conv8_stride = 1
    conv8_pad = "SAME"
    conv8_act = activation

    conv9_fmaps = 10
    conv9_ksize = 1
    conv9_stride = 1
    conv9_pad = "SAME"
    conv9_act = activation

    gap10_psize = 6
    gap10_stride = 1
    gap10_pad = "VALID"

    with tf.name_scope("input_dropout"):
        drop1 = tf.layers.dropout(X,
                                  rate=0.2,
                                  training=True,
                                  name="drop1")

    with tf.name_scope("block1"):
        conv1 = tf.layers.conv2d(drop1,
                                 filters=conv1_fmaps,
                                 kernel_size=conv1_ksize,
                                 strides=conv1_stride,
                                 padding=conv1_pad,
                                 activation=conv1_act,
                                 kernel_initializer=kernel_init,
                                 bias_initializer=bias_init,
                                 kernel_regularizer=regularizer,
                                 bias_regularizer=regularizer,
                                 name="conv1")

        conv2 = tf.layers.conv2d(conv1,
                                 filters=conv1_fmaps,
                                 kernel_size=conv2_ksize,
                                 strides=conv2_stride,
                                 padding=conv2_pad,
                                 activation=conv2_act,
                                 kernel_initializer=kernel_init,
                                 bias_initializer=bias_init,
                                 kernel_regularizer=regularizer,
                                 bias_regularizer=regularizer,
                                 name="conv2")

        conv3 = tf.layers.conv2d(conv2,
                                 filters=conv3_fmaps,
                                 kernel_size=conv3_ksize,
                                 strides=conv3_stride,
                                 padding=conv3_pad,
                                 activation=conv3_act,
                                 kernel_initializer=kernel_init,
                                 bias_initializer=bias_init,
                                 kernel_regularizer=regularizer,
                                 bias_regularizer=regularizer,
                                 name="conv3")

        drop2 = tf.layers.dropout(conv3,
                                  rate=0.5,
                                  training=True,
                                  name="drop2")

    with tf.name_scope("block2"):
        conv4 = tf.layers.conv2d(drop2,
                                 filters=conv4_fmaps,
                                 kernel_size=conv4_ksize,
                                 strides=conv4_stride,
                                 padding=conv4_pad,
                                 activation=conv4_act,
                                 kernel_initializer=kernel_init,
                                 bias_initializer=bias_init,
                                 kernel_regularizer=regularizer,
                                 bias_regularizer=regularizer,
                                 name="conv4")

        conv5 = tf.layers.conv2d(conv4,
                                 filters=conv5_fmaps,
                                 kernel_size=conv5_ksize,
                                 strides=conv5_stride,
                                 padding=conv5_pad,
                                 activation=conv5_act,
                                 kernel_initializer=kernel_init,
                                 bias_initializer=bias_init,
                                 kernel_regularizer=regularizer,
                                 bias_regularizer=regularizer,
                                 name="conv5")

        conv6 = tf.layers.conv2d(conv5,
                                 filters=conv6_fmaps,
                                 kernel_size=conv6_ksize,
                                 strides=conv6_stride,
                                 padding=conv6_pad,
                                 activation=conv6_act,
                                 kernel_initializer=kernel_init,
                                 bias_initializer=bias_init,
                                 kernel_regularizer=regularizer,
                                 bias_regularizer=regularizer,
                                 name="conv6")

        drop3 = tf.layers.dropout(conv6,
                                  rate=0.5,
                                  training=True,
                                  name="drop3")


    with tf.name_scope("block3"):
        conv7 = tf.layers.conv2d(drop3,
                                 filters=conv7_fmaps,
                                 kernel_size=conv7_ksize,
                                 strides=conv7_stride,
                                 padding=conv7_pad,
                                 activation=conv7_act,
                                 kernel_initializer=kernel_init,
                                 bias_initializer=bias_init,
                                 kernel_regularizer=regularizer,
                                 bias_regularizer=regularizer,
                                 name="conv7")

        conv8 = tf.layers.conv2d(conv7,
                                 filters=conv8_fmaps,
                                 kernel_size=conv8_ksize,
                                 strides=conv8_stride,
                                 padding=conv8_pad,
                                 activation=conv8_act,
                                 kernel_initializer=kernel_init,
                                 bias_initializer=bias_init,
                                 kernel_regularizer=regularizer,
                                 bias_regularizer=regularizer,
                                 name="conv8")

        conv9 = tf.layers.conv2d(conv8,
                                 filters=conv9_fmaps,
                                 kernel_size=conv9_ksize,
                                 strides=conv9_stride,
                                 padding=conv9_pad,
                                 activation=conv9_act,
                                 kernel_initializer=kernel_init,
                                 bias_initializer=bias_init,
                                 kernel_regularizer=regularizer,
                                 bias_regularizer=regularizer,
                                 name="conv9")

    with tf.name_scope("gap"):
        gap10 = tf.layers.average_pooling2d(conv9,
                                            pool_size=gap10_psize,
                                            strides=gap10_stride,
                                            padding=gap10_pad,
                                            name="gap10")

        logits = tf.reshape(gap10, shape=[-1, n_outputs], name="flatten")

    return logits

logits = AllCNN(X_reshaped, y, regularizer=regularizer, kernel_init=kernel_init, bias_init=bias_init, activation=activation)

with tf.name_scope("probs"):
    y_prob = tf.nn.softmax(logits, name="Y_prob")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    true_loss = tf.reduce_mean(xentropy)
    reg_loss = reg_constant*tf.losses.get_regularization_loss()
    loss = true_loss + reg_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(y_prob, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope('tensorboard') as scope:
    lossSummary = tf.summary.scalar('Loss', loss)
    accuracySummary = tf.summary.scalar('Accuracy', accuracy)
    fileWriter = tf.summary.FileWriter(logDir, tf.get_default_graph())

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

S_curr = 0
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for ctr, (X_batch, y_batch) in enumerate(shuffle_batch(X_train, y_train, batch_size)):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            #if ctr%100 == 0:
            #    xentropy_result, reg_loss_result, true_loss_result, loss_result, logits_result, y_prob_result, correct_result, accuracy_result = sess.run([xentropy, reg_loss, true_loss, loss, logits, y_prob, correct, accuracy], feed_dict = {X: X_batch, y: y_batch})
            #    pred_cat = [np.where(y_prob_result[i, :] == np.max(y_prob_result[i, :]))[0][0] for i in range(100)]
            #    for i in range(100):
            #        print("pred_cat: ", pred_cat[i], "true_cat: ", y_batch[i])
            #    print("loss: ", loss_result)
            #    pdb.set_trace()
        #xentropy_result, reg_loss_result, true_loss_result, loss_result, logits_result, y_prob_result, correct_result, accuracy_result = sess.run([xentropy, reg_loss, true_loss, loss, logits, y_prob, correct, accuracy], feed_dict = {X: X_batch, y: y_batch})
        loss_batch = loss.eval(feed_dict={X: X_batch, y: y_batch})
        loss_valid = loss.eval(feed_dict={X: X_valid, y: y_valid})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print("Epoch: %d; Train loss: %+4.3e; Train acc: %+3.2e; Valid loss: %+4.3e; Valid acc: %+3.2e"%(epoch, loss_batch, acc_batch, loss_valid, acc_valid))
        #pdb.set_trace()
        save_path = saver.save(sess, "./allCNN_model")
        lossSummaryStr = lossSummary.eval(feed_dict={X: X_batch, y: y_batch})
        accuracySummaryStr = accuracySummary.eval(feed_dict={X: X_batch, y: y_batch})
        fileWriter.add_summary(lossSummaryStr, epoch)
        fileWriter.add_summary(accuracySummaryStr, epoch)
    loss_test = loss.eval(feed_dict={X: X_test, y: y_test})
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Epoch: %d; Train loss: %+4.3e; Train acc: %+3.2e; Test loss: %+4.3e; Test acc: %+3.2e"%(epoch, loss_batch, acc_batch, loss_test, acc_test))
fileWriter.close()
