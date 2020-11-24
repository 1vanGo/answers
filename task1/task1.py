"""Solution for the 1st task."""

def multiply(a):
    """Multiplies array."""

    def dot(a):
        """Helper function."""
        res = 1
        for i in a:
            res *= i
        return res

    zero_index, res_list = [], []

    for n_i, i in enumerate(a):
        if i == 0:
            zero_index.append(n_i)
        if len(zero_index) > 1:
            res_list = [0 for j in range(len(a))]
            break

    if len(zero_index) == 0:
        dot_all = dot(a)
        res_list = [dot_all//i for i in a]
    elif len(zero_index) == 1:
        res_list = [0 for i in range(len(a))]
        ac = a.copy()
        ac.remove(0)
        res_list[zero_index[0]] = dot(ac)
    return res_list
