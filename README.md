# C-NN
Neural network library written in plain C for Dense and Convolutional nets.

MNIST_DENSE.c and MNIST_CONV.c are test files you can use to try the library out.
You can obtain MNIST dataset from http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip. Files have to be in .txt format to be correctly read. You also have to set the file path in MNIST_DENSE.c and MNIST_CONV.c. 

You can compile with the following commands:
gcc MNIST_DENSE.c net.c array.c layers.c data.c tensor.c -Ofast
gcc MNIST_CONV.c net.c array.c layers.c data.c tensor.c -Ofast

BLAS has to be installed. On MacOs, you can just compile with the additional flag "-framework Accelerate". On other systems you can install BLAS and link it with "-lblas" flag. 
