import os


def _localhost():
    """
    Use for debug ONLY
    :return:
    """
    os.system("python -m Pyro4.naming -n localhost -k asdf")


if __name__ == '__main__':
    _localhost()
