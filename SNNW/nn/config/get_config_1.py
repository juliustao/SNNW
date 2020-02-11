mnist_config_1 = """# 0
[fully_connected]
in_size=784
out_size=1000
activation=relu

# 1
[fully_connected]
in_size=1000
out_size=1000
activation=relu

# 2
[fully_connected]
in_size=1000
out_size=10
activation=softmax
"""


def get_config_1(config_path):
    with open(config_path, 'w') as f:
        f.write(mnist_config_1)
