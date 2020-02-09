curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O
curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O
curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O
curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O

gunzip t*-ubyte.gz

python convert_mnist_to_jpg.py . .
