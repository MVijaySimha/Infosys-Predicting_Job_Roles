def factorial_tail(n, acc=1):
    if n == 0 or n == 1:
        return acc
    return factorial_tail(n - 1, acc * n)
print("Tail Recursive:", factorial_tail(7))  



