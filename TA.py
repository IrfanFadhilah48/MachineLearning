import tensorflow as tf #untuk tensorflow
import cv2 #untuk opencv(image processing)
import os #untuk ambil file dari local disk
import numpy as np #untuk matriks
import matplotlib.pyplot as plt 
import random #deklarasi untuk random sample

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
    if os.path.isdir(os.path.join(data_directory, d))]

    labels = [] #merupakan labels yang berarray
    images = [] #merupakan images yang berarray

    for d in directories:
        label_directory = os.path.join(data_directory,d)
        file_names = [os.path.join(label_directory, f)
    for f in os.listdir(label_directory)
        if f.endswith(".png")]  #format yang dapat dibaca berformat .png saja
        for f in file_names:
            images.append(cv2.imread(f))
            labels.append(int(d))
    return images, labels

def neural_net(x):
    layer_1 = tf.layers.dense(x, 450, activation=tf.nn.relu) #inputan layer x, banyaknya neuron 450 unit, menggunakan fungsi aktivasi relu . Relu (Rectified LinearUnit)
    layer_2 = tf.layers.dense(layer_1, 200, activation=tf.nn.relu) #inputan dari layer 1 , neuronya sebanyak 200 lalu menggunakan fungsi aktivasi relu
    
    out_layer = tf.layers.dense(layer_2, 26) #inputan berasal dari layer 2 dan neuronya sebanyak 26 unit
    return out_layer #menampilakan hasil out layer

ROOT_PATH = "D:\Python L\Python Library + Dataset" #Direktori untuk library yang akan dibaca

train_data_directory = os.path.join(ROOT_PATH, "Place") #Nama folder yang berisi library

images, labels = load_data(train_data_directory) #untuk meload data library


dim = (30,30) #dimensi dari gambar yang ada di library

images30 = []

for image in images:
 gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 images30.append(cv2.resize(gray_image, dim, 
  interpolation=cv2.INTER_CUBIC))


x = tf.placeholder(dtype = tf.float32, shape = [None, 30, 30]) # fungsinya buat wadah simpan  bertipe float, simpen hasil gambar yang sudah diubah ke array
y = tf.placeholder(dtype = tf.int32, shape = [None]) # fungnsinya buat bikin wadah yang bertipe integer , simpen hasil dari label

images_flat = tf.contrib.layers.flatten(x) # fungnsinya array 1 dimensi ini dijadikan layar input akan lari ke baris 24

logits = neural_net(images_flat)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
 logits = logits, labels = y)) # digunakan sebagai untuk mengetahui baik buruknya hasil program bekerja, semakin kecil nilainya berati programnya semakin bagus

train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss) #learning rate terlalu kecil jika melihat inputan baru, dia ttp ingat inputan lama

correct_pred = tf.argmax(logits, 1) # untuk simpen benar

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) #untuk simpen akurasi belajar program

tf.set_random_seed(1234) # untuk mengambil hasil inputan secara acak 

with tf.Session() as sess:
 sess.run(tf.global_variables_initializer())
 for i in range(1200): # mengulang sebanyak 450 gambar secara acak, jika learning rate terlalu kecil si jumlah inputannya perbesar tanpa ada maksimal
  _, loss_value, acc = sess.run([train_op, loss, accuracy], feed_dict={x:images30, y:labels})
#acc diatas adalah akurasi. diatas menjalankan fungsinoptimizer, fungsi loss, fungsi akurasi
# feed untuk sumber inputnya , x untuk wadah gambar, y itu label dari gambarnya

  if i % 10 == 0:
   print("Loss: ", loss_value, "Accuracy: ", acc) #putaran kelipatan 10, loss value berapa dan akurasinya berapa

 sample_indexes = random.sample(range(len(images30)), 10) #mengambil gambar di library sebanyak 10 sample secara random..Kenapa random? karena sudah dideklarasikan random
 sample_images = [images30[i] for i in sample_indexes] #index yg udh di deklarasiin diatas, diambil 10 secara random
 sample_labels = [labels[i] for i in sample_indexes] #untuk nama folder digambar sebanyak 10 buah juga

 predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0] #ini untuk machine learning nebak (menentukan hasil prediksi) dengan menggunakan index sample images (karena hasilnya array dan = 0 makanya di tambah akhiran [0])

 string_labels = ['Gereja','College','eiffel','liberty','mosque','mountain','museum'] # untuk simpen nama nama dari si gambar dari si user

 fig = plt.figure(figsize=(10, 10)) #bikin gambar berukuran 10 x 10 , gambarnya diambil dari sample images
 for i in range(len(sample_images)): #perulangan index
    truth = sample_labels[i] #sample label dimasukin ke variabel truth (label asli) 
    prediction = predicted[i] #predicted dimasukin ke prediction (untuk label prediksinya )
    plt.subplot(5, 2,1+i) #bikin plot buat si gambar yang terdiri dari 2 kolom dan 5 baris( banyak data dalam 1 window)
    plt.axis('off') #mematikan grafik x dan y 
    color='green' if truth == prediction else 'red' # kalo jwb prediksi bener akan warna hijau, jika prediksi salah akan berwarna merah
    plt.text(40, 10, "Label Asli: {0}\nHasil Prediksi: {1}".format(string_labels[truth], string_labels[prediction]), #(0 sama 1 itu untuk menghitung variabel berdasarkan format string labelnya sendiri)
             fontsize=12, color=color) # 40, 10 untuk koordinatnya
    plt.imshow(sample_images[i]) #untuk nampilin hasil labelnya


 plt.show() #dia akan munculin hasil akhir keseluruhannya
