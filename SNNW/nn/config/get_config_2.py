mnist_config_2 = """# 0
[fully_connected]
in_size=784
out_size=1000
activation=tanh

# 1
[fully_connected]
in_size=1000
out_size=1000
activation=tanh

# 2
[fully_connected]
in_size=1000
out_size=10
activation=softmax
"""


def get_config_2(config_path):
    with open(config_path, 'w') as f:
        f.write(mnist_config_2)

