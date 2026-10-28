"""Compiled execution; model, loss, optimizer and batch order are unchanged."""
def make_train_step(tf, model, optimizer, loss_terms, sparse_lambda, compiled=True):
    def step(inputs, target, weights):
        with tf.GradientTape() as tape:
            prediction = model(inputs, training=True)
            terms = loss_terms(model, target, prediction, weights, sparse_lambda)
        gradients = tape.gradient(terms["total"], model.trainable_variables)
        pairs = []
        for gradient, variable in zip(gradients, model.trainable_variables):
            if gradient is not None:
                tf.debugging.assert_all_finite(tf.convert_to_tensor(gradient), "Nonfinite gradient")
                pairs.append((gradient, variable))
        optimizer.apply_gradients(pairs)
        return terms
    return tf.function(step, reduce_retracing=True) if compiled else step
