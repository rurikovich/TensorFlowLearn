from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf

# Launch the graph in a session.
with tf.compat.v1.Session() as ses:
    # Build a graph.
    a = tf.constant(5.0)
    b = tf.constant(60.0)
    c = a * b

    # Evaluate the tensor `c`.
    print(ses.run(c))
