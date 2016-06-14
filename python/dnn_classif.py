#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# 
# Copyright 2016 <+YOU OR YOUR COMPANY+>.
# 
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

import tensorflow as tf
import numpy as np
import pmt
from gnuradio import gr

class dnn_classif(gr.basic_block):
    """
    docstring for block dnn_classif
    """
    def __init__(self, path):
        gr.basic_block.__init__(self,
            name="dnn_classif",
            in_sig=[(np.float32,512)],
            out_sig=None)

        self.message_port_register_out(pmt.intern('out'))

        self.x = tf.placeholder(tf.float32,shape=(1,512))
        self.labels = [ 'dpsk','qpsk','8psk', 'ssb','dsb','nbfm','wfm','qam16','qam64','gfsk','ofdm']
        wts1  = np.loadtxt(path+'/weights1')
        bs1   =  np.loadtxt(path+'/biases1')
        wts2  = np.loadtxt(path+'/weights2')
        bs2   =  np.loadtxt(path+'/biases2')
        wts3  = np.loadtxt(path+'/weights3')
        bs3   =  np.loadtxt(path+'/biases3')
        self.weights1 = tf.constant(wts1,dtype="float32")
        self.biases1= tf.constant(bs1,dtype="float32")
        self.weights2 = tf.constant(wts2,dtype="float32")
        self.biases2= tf.constant(bs2,dtype="float32")
        self.weights3 = tf.constant(wts3,dtype="float32")
        self.biases3= tf.constant(bs3,dtype="float32")
        self.sess = tf.Session()
        self.op = tf.nn.softmax(tf.matmul(tf.nn.sigmoid(tf.matmul(tf.nn.sigmoid(tf.matmul(self.x, self.weights1) + self.biases1),self.weights2)+self.biases2),self.weights3)+self.biases3)

    def forecast(self, noutput_items, ninput_items_required):
        #setup size of input_items[i] for work call
        for i in range(len(ninput_items_required)):
            ninput_items_required[i] = noutput_items

    def post_msg(self, label):
        self.message_port_pub(pmt.intern('out'), pmt.cons(pmt.PMT_NIL, pmt.intern(label)))

    def general_work(self, input_items, output_items):
        in0 = np.array(input_items[0])
        for i in range(in0.shape[0]):
            inrow = in0[i]
            soft = self.sess.run([self.op], feed_dict={self.x:np.reshape(inrow,(1,512))})
            label = np.argmax(soft[0],1)
            #print  "Class:",self.labels[label]
            self.post_msg(self.labels[label])
        self.consume_each(len(input_items[0]))
        return len(input_items[0])
