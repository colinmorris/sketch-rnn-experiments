import random
import numpy as np
from magenta.models.sketch_rnn import utils

def sample_completions(sess, model, partial,
    seq_len=250, temperature=1.0, greedy_mode=False,
           z=None):
  """Starting from the given vector of strokes, sample a complete sketch
  that uses that prefix.
  ASSUMPTION: partial begins with standard dummy starter stroke
  """

  def adjust_temp(pi_pdf, temp):
    pi_pdf = np.log(pi_pdf) / temp
    pi_pdf -= pi_pdf.max()
    pi_pdf = np.exp(pi_pdf)
    pi_pdf /= pi_pdf.sum()
    return pi_pdf

  def get_pi_idx(x, pdf, temp=1.0, greedy=False):
    """Samples from a pdf, optionally greedily."""
    if greedy:
      return np.argmax(pdf)
    pdf = adjust_temp(np.copy(pdf), temp)
    accumulate = 0
    for i in range(0, pdf.size):
      accumulate += pdf[i]
      if accumulate >= x:
        return i
    tf.logging.info('Error with sampling ensemble.')
    return -1

  def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
    if greedy:
      return mu1, mu2
    mean = [mu1, mu2]
    s1 *= temp * temp
    s2 *= temp * temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]
    
  start_stroke_token = [0, 0, 1, 0, 0]

  nstrokes, ndims = partial.shape
  assert ndims == 3, 'Need partials in 3/5-stroke format, got shape'.format(partial.shape)
  if ndims == 3:
    partial = utils.to_big_strokes(partial)
    assert (partial[0] != start_stroke_token).any()
    partial = np.concatenate([ [start_stroke_token],  partial ])
    nstrokes += 1
  assert nstrokes < seq_len
  assert nstrokes > 0
  assert (partial[0] == start_stroke_token).all(), partial[0]
  
  prev_x = partial[0].reshape(1,1,5)
  if z is None:
    z = np.random.randn(1, model.hps.z_size)  # not used if unconditional

  if not model.hps.conditional:
    prev_state = sess.run(model.initial_state)
  else:
    prev_state = sess.run(model.initial_state, feed_dict={model.batch_z: z})

  #strokes = np.zeros((seq_len, 5), dtype=np.float32)
  # I am guilty of off-by-one mismanagement
  #strokes[:nstrokes-1] = partial[1:]
  strokes = np.copy(partial)
  
  #mixture_params = []
  greedy = False
  temp = 1.0

  for i in range(seq_len):
    # update the hidden state based on the previous state and the ith stroke
    # if there is no i+1th stroke in our prefix, sample one, add to strokes,
    # and save it as input the next time around.
    if not model.hps.conditional:
      feed = {
          model.input_x: prev_x,
          model.sequence_lengths: [1],
          model.initial_state: prev_state
      }
    else:
      feed = {
          model.input_x: prev_x,
          model.sequence_lengths: [1],
          model.initial_state: prev_state,
          model.batch_z: z
      }

    params = sess.run([
        model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2, model.corr,
        model.pen, model.final_state
    ], feed)

    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, next_state] = params

    if i < 0: # wtf is going on?
      greedy = False
      temp = 1.0
    else:
      greedy = greedy_mode
      temp = temperature

    # update hidden state no matter what
    prev_state = next_state
    
    if i >= nstrokes-1:
      # we don't know what the i+1th stroke is supposed to be, so sample one
      idx = get_pi_idx(random.random(), o_pi[0], temp, greedy)

      idx_eos = get_pi_idx(random.random(), o_pen[0], temp, greedy)
      eos = [0, 0, 0]
      eos[idx_eos] = 1

      next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
                                            o_sigma1[0][idx], o_sigma2[0][idx],
                                            o_corr[0][idx], np.sqrt(temp), greedy)

      strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

      params = [
          o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0],
          o_pen[0]
      ]

      #mixture_params.append(params)

      prev_x = np.zeros((1, 1, 5), dtype=np.float32)
      prev_x[0][0] = np.array(
          [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
    else:
      prev_x = strokes[i].reshape(1,1,5) #partial[i+1]
      
  return strokes
  #, mixture_params
