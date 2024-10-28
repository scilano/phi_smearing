from time import perf_counter
import coefficient_eval
import numpy as np

t1 = perf_counter()
a = coefficient_eval.I1_eval([1e-2, 1e-2, 1e-2], 1, 1, 1, np.log(10) * 3)
t2 = perf_counter()
b = coefficient_eval.I1_eval([1e-2, 1e-2, 1e-2], 1, 1, 1, 5)
t3 = perf_counter()
c = coefficient_eval.I1_eval_old([1e-2, 1e-2, 1e-2], 1, 1, 1)
t4 = perf_counter()
print("method1")
print(t2 - t1)
print("method2")
print(t4 - t3)
print(a, b, c)
