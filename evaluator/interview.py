import numpy as np

def count_cost(M):
    m = M.shape[0]
    n = M.shape[1]
    dp = np.array(m + n + 1)
    dp[0] = 0
    m_init = 0
    n_init = 0
    for i in range(m + n):
        if M[m_init + 1][n_init] >= M[m_init][n_init +1]:
            dp[i] = M[m_init][n_init +1]
            n_init += 1
        else:
            dp[i] = M[m_init + 1][n_init]
            m_init += 1

    print(dp[[m+n+1]])