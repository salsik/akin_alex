import os
import time


import numpy as np
import tensorflow as tf


from datetime import datetime

from pathlib import Path

import re




def train(self):
    print("Start reading")
    print("epoch: {}".format(self.epoch))

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    #summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    #file_writer.set_as_default()
    #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


    with file_writer.as_default():
        for epoch in range(self.epoch_start, self.epoch_start + self.epoch):
            epoch_start_time = time.time()

            for image_batch in self.data_generator.dataset:
                
                # just to sisplay the image  afterdecoding and resizing
                #self.plt_display(image_batch[0][0],"dsds")
                z = tf.random.truncated_normal(shape=(self.batch_size, self.z_dim), dtype=tf.float32)
                c = tf.random.uniform([self.batch_size, 1], minval=0, maxval=4, dtype=tf.dtypes.int32)
                d_loss, gp_loss = self.train_discriminator_step(image_batch, z)
                g_loss = self.train_generator_step(z, c)

            if (epoch % self.print_freq) == 0:
                print(
                    "epoch {}/{} ({:.2f} sec):, d_loss {:.4f}, gp_loss {:.4f}, g_loss {:.4f}".format(
                        epoch,
                        self.epoch_start + self.epoch,
                        time.time() - epoch_start_time,
                        d_loss.numpy(),
                        gp_loss.numpy(),
                        g_loss.numpy(),
                    )
                )
                tf.summary.scalar('DL loss', d_loss.numpy(),step=epoch)
                tf.summary.scalar('GP loss', gp_loss.numpy(),step=epoch)
                tf.summary.scalar('G loss', g_loss.numpy(),step=epoch)
                #merged_summary_op = tf.summary.merge_all()

                file_writer.flush()
                #file_writer.add_summary(tf.summary.text, epoch )

            if (epoch % self.save_freq) == 0:
                self.save_models(epoch)
                self.save_samples(epoch)


def read_logs():

    #filename = '/data1/data_alex/rico/akin experiments/exp3 tanh lr 0.0005 0.0008 0.2 0.9 0.5/terminal_output.txt'
    filename = '/data1/data_alex/rico/akin experiments/exp1 relu default lr/readme.txt'

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

 
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")


    with file_writer.as_default():


        with open(filename) as file:
            for line in file:
                if (line.startswith("epoch ")):

                    print(line.rstrip())
                    #s = 'asdf=5;iwantthis123jasd'
                    epoch = re.search('epoch(.*)/5000', line)
                    d_loss = re.search('d_loss (.*), gp_loss', line)
                    gp_loss = re.search('gp_loss (.*), g_loss', line)
                    g_loss = re.search('g_loss (.*)', line)

                    ll = np.array([epoch.group(1),d_loss.group(1),gp_loss.group(1),g_loss.group(1)])
                    #flot_arr = ll.astype(np.float)
                    flot_arr = np.asarray(ll, dtype=float)
                    print(epoch.group(1),d_loss.group(1),gp_loss.group(1),g_loss.group(1))
                    tf.summary.scalar('DL loss', flot_arr[1],step=flot_arr[0])
                    tf.summary.scalar('GP loss', flot_arr[2],step=flot_arr[0])
                    tf.summary.scalar('G loss',flot_arr[3],step=flot_arr[0])
                

                    file_writer.flush()
                 


def main():
    
    read_logs()


if __name__ == "__main__":
    main()