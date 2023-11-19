# -*- coding: utf-8 -*-
# @Time    : 2023/11/19 13:25
# @FileName: time_tools.py

import time

def timeit(method):
    def wrapper(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        print(f"Execution time: {'%.4f' % (end_time - start_time)}")
        return result
    return wrapper

if __name__ == "__main__":
    @timeit
    def test_function(a, b):
        return a+b
    
    
    test_function(1, 2)


        