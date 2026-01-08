# %% [markdown]
# # Singular Value Decomposition
# 
# - Bui Cao Tri, University Of Information Technology
# - Ho Quoc Trung, University Of Information Technology
# - Pham Tien, University Of Information Technology
# - Phan Cong Thuan, University Of Information Technology
# 

# %% [markdown]
# When approaching real-world problems, we are often faced with massive datasets. However, real-world data is frequently messy, redundant, and too large to process efficiently in its raw form. This creates a critical need for dimensionality reduction algorithms.
# 
# In this file, we will examine exactly how Singular Value Decomposition (SVD) transforms data. We will walk through the underlying mathematical formulas step-by-step, ensuring you understand the mechanics 'under the hood' rather than treating functions like `model.fit` as a black box.

# %% [markdown]
# #### Import libraries that will be needed
# We only use random for generating data. No numpy or sklearn is used for the core logic.

# %%
import random

# %% [markdown]
# ## Section 1: Foundational Linear Algebra

# %% [markdown]
# Before tackling SVD, we must establish the fundamental operations of our linear algebra engine. These functions replace standard libraries like Numpy for matrix manipulation.

# %% [markdown]
# ### 1.1 Matrix Utility & Arithmetic
# 

# %% [markdown]
# Basic tools for creating, viewing, and modifying matrix structures.
# 
# 
# - **generate matrix**: Creates random matrices for testing.
# 
# - **print matrix**: Visualizes the data rows and columns.
# 
# - **copy matrix**: Deep copies data to prevent side effects during manipulations.
# 
# - **scalar mul matrix**: Scaling a matrix by a constant.
# 
# - **sub matrix**: Element-wise subtraction ($A - B$).

# %%
def generate_matrix(rows, cols):
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(random.randint(0, 10)) 
        matrix.append(row)
    return matrix


# %%
def print_matrix(M, name):
    print(f"--- Matrix {name} ---")
    if not M or not M[0]: 
        print("Empty or Invalid Structure")
        return
    for row in M:
        print(["{:.4f}".format(x) for x in row])
    print()
    return

# %%
def copy(A):
    B=[]
    for i in range(len(A)):
        row = []
        for j in range(len(A[0])):
            row.append(A[i][j])
        B.append(row)
    return B

# %%
def scalar_mul(k, A):
    B = copy(A)
    for i in range(len(B)):
        for j in range(len(B[i])):
            B[i][j] *= k
    return B

# %%
def sub_matr(A, B):
    if ((len(A) != len (B)) | (len(A[0]) != len(B[0]))):
        print("Mismatch size between 2 matrixs")
        return
    C = copy(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = A[i][j]-B[i][j]
    return C

# %% [markdown]
# ### 1.2 Structural Transformations

# %% [markdown]
# Operations that change the shape or orientation of the data.
# 
# 
# - **Transpose**: Swapping rows and columns ($A^T$).
# 
# - **mul matrix**: Matrix-Matrix multiplication (The core $O(n^3)$ operation).

# %%
def T(A):
    B=[]
    for i in range(len(A[0])):
        row = []
        for j in range(len(A)):
            row.append(A[j][i])
        B.append(row)
    return B

def mul_matr(A, B):
    if len(A[0]) != len(B):
        print("Cannot multiply these two matrix cause of incorrect corresponding size")
        return
    result = []
    for _ in range(len(A)):
        vector_0 = []
        for _ in range(len(B[0])):
            vector_0.append(0)
        result.append(vector_0)
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(A[0])):
                result[i][j] += A[i][k]*B[k][j]
    return result

    

# %% [markdown]
# ## Section 2: Vector Space Operations

# %% [markdown]
# SVD relies heavily on the geometric properties of vectors (length, angles, and orthogonality).

# %% [markdown]
# ### 2.1 Measurements

# %% [markdown]
# - **dot vector**: The inner product, determining how much one vector goes in the direction of another. 
# 
# - **norm2 vector**: Euclidean length ($||v||_2$). 
# 
# - **cosine similarity**: Measuring the angle between vectors (independent of magnitude).

# %%
def dot_vec(A, B):
    if len(A) != len(B):
        print("Cannot multiply these two matrix cause of incorrect corresponding size")
        return
    result = 0
    for i in range(len(A)):
        result += A[i][0] * B[i][0]
    return result

def norm2_vec(A):
    result = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            result+=A[i][j]**2
    result = result ** 0.5
    return result

def cosine_similarity(A, B):
    normA = norm2_vec(A)
    normB = norm2_vec(B)
    if normA < 1e-16 or normB < 1e-16:
        return 0
    result = dot_vec(A, B)
    result /= normA*normB
    return result

# %% [markdown]
# ### 2.2 Orthogonalization & Normalization

# %% [markdown]
# - **normalize vector**: Scaling a vector to unit length ($||v||=1$).
# 
# - **gram-schmidt**: The process of orthonormalizing a set of vectors. This ensures our bases $U$ and $V$ remain orthogonal.

# %%
def normalize(v):
    norm = norm2_vec(v)
    if norm < 1e-16:
        return v
    v_normalize = []
    for i in range(len(v)):
        row = []
        for j in range(len(v[0])):
            row.append(v[i][j])
        v_normalize.append(row)
    for i in range(len(v)):
        for j in range(len(v[0])):
            v_normalize[i][j] /= norm
    return v_normalize

def gram_schmidt(v, eig_vec):
    if eig_vec == None:
        return v
    for t in range(len(eig_vec)):
        vec= copy(v)
        for i in range(len(vec)):
            vec[i][0] = eig_vec[t][i]
        v = sub_matr(v, scalar_mul(dot_vec(v, vec), vec))
    v = normalize(v)
    return v

# %% [markdown]
# ## Section 3: The Engine - Eigendecomposition

# %% [markdown]
# Since SVD is derived from the eigendecomposition of $A^TA$, we implement the iterative algorithms required to find eigenvalues and eigenvectors without using a closed-form solver.

# %% [markdown]
# ### 3.1 Power Iteration

# %% [markdown]
# - **find max eigen**: An iterative method to find the dominant eigenvector and eigenvalue (largest $\lambda$) of a matrix.

# %% [markdown]
# $$A = \lambda_1 v_1 v_1^T + \lambda_2 v_2 v_2^T + \dots + \lambda_n v_n v_n^T$$

# %%
def find_max_eig(A, eig_vec, max_iteration = 1000):   
    rand_vec = []
    n = len(A)
    for i in range(n):
        row = []
        row.append(random.uniform(-1, 1))
        rand_vec.append(row)
    if norm2_vec(rand_vec) == 0:
        rand_vec[0][0] = 1.0
    rand_vec = normalize(rand_vec)
    for i in range(max_iteration):
        temp_vec = mul_matr(A, rand_vec)
        temp_vec = normalize(temp_vec)
        temp_vec = gram_schmidt(temp_vec, eig_vec)
        if(1-abs(cosine_similarity(rand_vec, temp_vec)) < 1e-16):
            break
        rand_vec=temp_vec
    temp = mul_matr(A, temp_vec)
    e_val = mul_matr(T(temp_vec), temp)
    eig_val = e_val[0][0]
    return eig_val, temp_vec
    

# %% [markdown]
# ### 3.2 Deflation & Full Decomposition

# %% [markdown]
# - **calculate eigen**: Repeatedly applies Power Iteration and then "deflates" (removes) the found eigenvalue from the matrix to find the next one, eventually retrieving all eigen-pairs.

# %%
def cal_eig(A):
    eig_val = []
    eig_vec = []
    n = len(A)
    current_matrix = A
    for i in range(n):
        val, vec = find_max_eig(current_matrix, eig_vec)
        eig_val.append(val)
        eig_vec.append([vec[j][0] for j in range(len(vec))])
        current_matrix = sub_matr(current_matrix, scalar_mul(val, mul_matr(vec, T(vec))))
    return eig_val, eig_vec


# %% [markdown]
# ## Section 4: Singular Value Decomposition (SVD)

# %% [markdown]
# The culmination of all previous modules. We construct the decomposition $A = U \Sigma V^T$.

# %% [markdown]
# ### 4.1 The Main Algorithm

# %% [markdown]
# SVD:
# 1. Computes $A^TA$.
# 2. Extracts eigenvectors ($V$) and eigenvalues ($\lambda$).
# 3. Computes Singular Values $\sigma = \sqrt{\lambda}$.
# 4. Computes $U$ via the mapping $u_i = \frac{Av_i}{\sigma_i}$.

# %%
def SVD(A):
    n, m = len(A), len(A[0])
    ATA = mul_matr(T(A), A)
    eig_val, V = cal_eig(ATA)
    E = []
    for i in range(len(ATA)):
        row=[]
        for j in range(len(ATA)):
            if i != j:
                row.append(0)
            else:
                row.append(abs(eig_val[i])**0.5)
        E.append(row)
    U = []
    for i in range(len(V)):
        if(E[i][i]>1e-4):
            vi = []
            vi.append(V[i])
            row = scalar_mul(1/E[i][i], mul_matr(A, T(vi)))
            row = normalize(row)
            row1 = []
            for j in range(len(row)):
                row1.append(row[j][0])
            U.append(row1)
        else:
            row1 = []
            for j in range(n):
                row1.append(0)
            U.append(row1)
    return T(U), E, V

# %% [markdown]
# ## Section 5: Experiment & Validation

# %% [markdown]
# We generate a random matrix and verify that $A \approx U \Sigma V^T$.

# %%
if __name__ == "__main__":
    ROWS = 10
    COLS = 30
    A_big = generate_matrix(ROWS, COLS)

    print(f"--- TESTING WITH RANDOM {ROWS}x{COLS} MATRIX ---")
    print_matrix(A_big[:3], f"A Original (First 3 rows of {ROWS})") 

    print("Calculating SVD... (Please wait)...")
    U_res, E_res, V_res = SVD(A_big)

    print(f"Shape of U: {len(U_res)}x{len(U_res[0])} (Expect {ROWS}x{ROWS} or {ROWS}x{COLS})")
    print(f"Shape of Sigma: {len(E_res)}x{len(E_res[0])} (Expect {COLS}x{COLS} or similar)")
    print(f"Shape of V.T: {len(V_res)}x{len(V_res[0])} (Expect {COLS}x{COLS})")

    UE = mul_matr(U_res, E_res)
    VT = V_res
    A_reconstructed = mul_matr(UE, VT)

    print_matrix(A_reconstructed[:3], f"A Reconstructed (First 3 rows)")

    error = 0
    for i in range(len(A_big)):
        for j in range(len(A_big[0])):
            error += (A_big[i][j] - A_reconstructed[i][j])**2

    print(f"\n========================================")
    print(f"Total Reconstruction Error (MSE): {error:.15f}")
    print(f"========================================")

    if error < 1e-4:
        print("\n✅ SUCCESS! The From-Scratch SVD algorithm works correctly.")
    else:
        print("\n❌ FAILED. High reconstruction error.")

# %% [markdown]
# ## Conclusion

# %% [markdown]
# In this project, we successfully implemented Singular Value Decomposition entirely from scratch, relying only on the random library for data generation. By building every component—from basic matrix multiplication to the Power Iteration algorithm—we have demonstrated the internal mechanics that sophisticated libraries like Numpy or Scikit-learn abstract away.


