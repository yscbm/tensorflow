# -*- coding:utf-8 -*-  

import tensorflow as tf

def maybe_download(filepath,filename):
  if not tf.gfile.Exists(filepath):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.Size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath
