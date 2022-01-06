from multiprocessing import Process, Pool, Queue, Pipe, Lock
import os


def info(title):
    print(title)
    print(f'module name: {__name__}')
    print(f'parent process: {os.getppid()}')
    print(f'process id: {os.getpid()}')


def f(name):
    info('function f')
    print(f'hello {name}')


if __name__ == "__main__":
    info('main line')
    p = Process(target=f, args=('bob', ))
    p.start()
    p.join()
