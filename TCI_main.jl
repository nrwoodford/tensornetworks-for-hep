using LinearAlgebra
using ITensors
using ITensorMPS
using PrettyTables
using Plots
#using ImageMagick, FileIO
#using Images

# Define phase gate operator for qubits, parameterized by θ
ITensors.op(::OpName"Phase", ::SiteType"S=1/2", s::Index; theta=0.0) = diagITensor([1.0, exp(im*theta)], s, dag(s))

# All equations referenced in the code correspond to the following paper: https://scipost.org/10.21468/SciPostPhys.18.3.104

# Takis parameters:
# Define parameters and perform TCI
N = 8 # number of qudits in *each* spatial dimension - was 50
d = 2 # physical dimension of the qudits                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         # local degree of freedom dimension of the MPS
num_variables = 1 # number of spatial dimensions - or in integration this is just though of as the number of variables to integrate over
N_total = N * num_variables
min_grid = [-10 for _ in 1:num_variables] # defines the minimum of the grid in each spatial dimension
max_grid = [10 for _ in 1:num_variables] # defines the maximum of the grid in each spatial dimension
tolerance = 1e-14 # Tolerance on the absolute value of a pivot in a given LDU decomposition (algorithm stopping condition)
max_pivots = 100 # Maximum number of pivots for a given LDU decomposition (algorithm stopping condition)
sweeps = 15 # maximum number of back and forth TCI sweeps
num_starting_pivots = 10 # number of random pivots to initialize the pivots before starting the TCI - was 100
sites = ITensors.siteinds(d, N_total) # ITensors sites (see documentation: https://docs.itensor.org/ITensorMPS/stable/IncludedSiteTypes.html#SiteTypes-Included-with-ITensor)


function fixed_dinary_to_decimal(dinary::Vector{Int}, d::Int)
    
    """
    Convert a "dinary" vector to its corresponding decimal value.

    A dinary vector is a representation of a number in base `d`, using digits from `1` to `d`.
    This function adjusts for the offset (by subtracting 1 from each digit) to convert it into a standard base-`d` representation with digits in the range `0` to `d-1`, and computes the corresponding decimal value.

    Arguments
    - `dinary::Vector{Int}`: A vector of integers in the range `1` to `d`, representing a number in base `d`.
    - `d::Int`: The base of the numeral system (must be ≥ 2).

    Returns
    - decimal: The decimal value corresponding to the input dinary vector.

    Example
    ```julia
    fixed_dinary_to_decimal([1, 2, 1], 3) # returns 3
    
    Explanation: [1,2,1] → [0,1,0] → 0 x 3² + 1 x 3¹ + 0 x 3⁰ = 3
    """
    
    # Step 1: Check that the base d is valid (must be at least 2)
    if d < 2
        error("Invalid Input: d must be at least 2")
    end
    
    # Step 2: Initialize a variable decimal to hold the result. Also create a variable power that starts at the appropriate power of dbased on the length of the input vector (start from most significant digit)
    decimal = 0
    power = length(dinary)-1
    
    # Step 3: Convert the dinary digits from the range 1 to d into standard base-d digits by subtracting 1 from each digit
    dinary .-= 1
    
    # Step 4: Loop over each digit in the adjusted dinary vector.Inside the loop:

    # Check if the digit is in the valid range [0, d-1]. If not, raise an error.
    # Multiply the digit by d raised to the current power and add the result to decimal.
    # Decrease the power by 1 after each iteration.
    
    for digit in dinary
        if digit < d && digit >= 0
            decimal += digit * d^power
            power -= 1
        else
            error("Invalid Input: digit not in valid range [0, d-1]")
        end 
    end
    
    return decimal
end

function decimal_to_dinary(decimal::Int, N::Int, d::Int)
    """
    Converts a decimal number to a dinary string for indexing into the tensor and accessing function values

    Inputs:
    - decimal: Number which is to be converted
    - N: the number of sites 
    - d: dimensionality of the string
    
    Outputs:
    - dinary: a 1-indexed dinary string representing the decimal
    """
    @assert decimal < d^N  # check this number can be represented in a string of N d-dimensional bits
    @assert decimal >= 0 # check non-negative

    dinary = reverse(digits(decimal, base=d)) # convert to list
    dinary = vcat(fill(0, N - length(dinary)), dinary)  # left pad it with 0s so length N
    dinary .+= 1  # so its not 0-indexed

    return dinary
end

function tensor_contraction(A, c_A::Tuple, B, c_B::Tuple)::Array{ComplexF64}
    
    """
    The contraction function takes 2 tensors A, B and 2 tuples c_A, c_B and returns another tensor after contracting A and B.

    A: first tensor c_A: indices of A to contract (Tuple of Int64) B: second tensor c_B: indices of B to contract (Tuple of Int64)

    Note 1: c_A and c_B should be the same length and the first index from c_A should have the same dimension as the first index of c_B, and so on.

    Note 2: It is assumed that the first index in c_A is to be contracted with the first index in c_B and so on.

    Note 3: Tuples are used instead of vectors for performance reasons (immutability, memory, etc).

    Example: contraction(A, (1, 4), B, (2, 1))
    """
    
    # Step 1: Get dimensions of A and B
    dim_A = size(A)
    dim_B = size(B)
    
    # Step 2: Identify uncontracted indices
    u_A = setdiff(1:length(dim_A), c_A)
    u_B = setdiff(1:length(dim_B), c_B)
    
    # Step 3: Extract dimensions of contracted and uncontracted indices
    c_A_dims = [size(A, dim) for dim in c_A]
    c_B_dims = [size(B, dim) for dim in c_B]

    u_A_dims = [size(A, dim) for dim in u_A]
    u_B_dims = [size(B, dim) for dim in u_B]
    
    # Step 4: Validate that contracted dimensions match
    @assert(c_A_dims == c_B_dims)
    
    # Step 5: Permute A and B to move contracted indices to the end of A and start of B
    A = permutedims(A, (u_A..., c_A...))
    B = permutedims(B, (c_B..., u_B...))
    
    # Step 6: Reshape A and B into matrices suitable for multiplication      
    A = reshape(A, prod(u_A_dims), prod(c_A_dims))
    B = reshape(B, prod(c_B_dims), prod(u_B_dims))
    
    # Step 7: Perform matrix multiplication    
    C = A*B
    
    # Step 8: Reshape result to uncontracted dimensions   
    C = reshape(C, (u_A_dims...,u_B_dims...))
    
    return C
end

function lu_full_pivoting(A, max_pivots, tolerance, all_rows, all_cols)
    
    """
    Perform LDU factorization with full pivoting of matrix A such that:
    A = transpose(P) * L * D * U * transpose(Q)

    # Inputs
    - A: matrix to factorize (ComplexF64)
    - max_pivots: max number of pivots (rank truncation)
    - tolerance: threshold for pivot magnitude to stop
    - all_rows: list of multi-index row labels of A
    - all_cols: list of multi-index column labels of A

    # Returns
    - P: permutation matrix for row swaps
    - L_truncated, D_truncated, U_truncated: L, D, U matrices truncated based on pivot rank or tolerance
    - Q: permutation matrix for column swaps
    - new_row_pivots: updated row multi-indices after pivoting
    - new_col_pivots: updated col multi-indices after pivoting
    - pivot_error: estimated error of the factorization
    """


    n,m = size(A)

    L = Matrix{Float64}(I, n, n)  # nxn identity matrix
    U = A  # Copy of A
    P = Matrix{Float64}(I, n, n)  # Another nxn identity#
    Q = Matrix{Float64}(I, m, m)  # mxm identity matrix

    new_row_pivots = []  # initialise empty list
    new_col_pivots = []

    pivot_error = 0
    col = 1  # column pointer
    pivots = 0
    max_rank = min(max_pivots, n, m)

    while pivots < max_rank
    
        pivot = maximum(abs, U[col:end, col:end])  # Find the pivot and its location
        pivot_row, pivot_col = Tuple(argmax(abs.(U[col:end, col:end]))).+col.-1  # need to change the location to account for submatrix

        # End if the pivot size becomes less than the tolerence and then we will truncate from here
        pivot_error = pivot
        if abs(pivot)^2 < tolerance
            break
        end

    
        push!(new_row_pivots, all_rows[pivot_row])  # Record the pivot row - its multi-index!
    
        if pivot_row != col
            # need to swap the rows in U, P, L and update all_rows
            U[[col, pivot_row], :] = U[[pivot_row, col], :]  # pivot now at location (col,col)
            P[[col, pivot_row], :] = P[[pivot_row, col], :]

            if col > 1  # onyl want to swap the values of L below the diagonal
                L[[col, pivot_row], 1:col-1] = L[[pivot_row, col], 1:col-1] 
            end
        
            all_rows[[col, pivot_row]] = all_rows[[pivot_row, col]]  # Swap the multi-indices in all_rows
        end

        push!(new_col_pivots, all_cols[pivot_col])  # Record pivot column
    
        if pivot_col != col
            # need to swap the cols in U, Q and update all_cols
            U[:, [col, pivot_col]] = U[:, [pivot_col, col]]
            Q[: ,[col, pivot_col]] = Q[:, [pivot_col, col]]
            all_cols[[col, pivot_col]] = all_cols[[pivot_col, col]]  # Swap the multi-indices in all_cols
        end

        # Perform Gaussian elimination to step L and U towards lower and upper triangular matrices
        m_factors = U[col+1:end, col] ./ U[col, col]  # Vector holding multiplicative factors
        L[col+1:end, col] = m_factors  # Update L
        U[col+1:end, :] .-= m_factors * transpose(U[col, :])  # Update U

        # Increment pivots and column
        pivots += 1
        col += 1

    end

    # Truncate L and U 
    L_truncated = L[:, 1:col-1]
    U_truncated = U[1:col-1, :]

    # Update pivot error to 0 if not contraction
    if pivots == min(n,m)
        pivot_error = 0  
    end

    diagonal = vcat(diag(U_truncated), ones(max(0, size(U_truncated, 1) - length(diag(U_truncated)))))  # Need to add ones to diagonal if rank A not min(n,m)
    D_truncated = Matrix(Diagonal(diagonal))
    inv_D = Matrix(Diagonal(diagonal.^-1))  
    U_truncated = inv_D*U_truncated  # Update U to have 1s along diagonal by dividing rows by elements of D

    return P, L_truncated, D_truncated, U_truncated, Q, new_row_pivots, new_col_pivots, pivot_error
end

function func(dinary, min_grid, max_grid, d, num_variables, N)

    """
    Convert a dinary vector to a grid value and then evaluate a given function with that input value.

    # Arguments
    - `dinary::Vector{Int}`: A vector of integers between 1 and `d`, e.g., [3, 2, 3, 1] for d = 3.
    - `min_grid::Vector{Float64}`: The minimum bounds of the grid in each dimension.
    - `max_grid::Vector{Float64}`: The maximum bounds of the grid in each dimension.
    - `d::Int`: The base used in the dinary representation.
    - `num_variables::Int`: The number of variables in the input space (used to stride through `dinary`). e.g for integration
    - `N::Int`: The dimensionality of the grid (used in calculating how many total points there are). number of point in each dimension

    # Returns
    - `x::Float64`: The computed grid value from the dinary vector.
    - `y::Float64`: The value of `sin(x)` evaluated at the computed grid value.
    """
    
    # Step 1: Calculate the spacing between points on the grid (delta_x).
    delta_x = (max_grid .- min_grid)./(d^N-1)  # d^N-1
    
    # Step 2: Convert the corresponding dinary digits for each spatial dimension to a decimal value using fixed_dinary_to_decimal()
    digits  = length(dinary)/num_variables  # number of dinary digits for each dimension of the grid

    # dinarys_list = reshape(dinary, (digits,num_variables))  # reshape dinary vector into a list of the dinarys for each dimension - can choose differrent groupings
    dinarys_list = [dinary[i:num_variables:end] for i in 1:num_variables] # or reshape locally - more commonly useful e.g. 2 variables, every other elements assigned to each list

    decimals_list = [fixed_dinary_to_decimal(dinary, d) for dinary in dinarys_list]  # convert these dinarys to a vector of decimal_list


    # Step 3: Compute the corresponding grid value.
    x = min_grid .+ decimals_list.*delta_x  # starting from the grid minimum, add on the decimal value times by the spacing to get a vector of coords
    
    #Step 4: Evaluate the function at the computed grid value.
    y = prod((sin.(5x) + 0.5cos.(3x.^2)).*exp.(-0.1x.^2) + atan.(x)/5 + 0.1.*x.*cos.(x))  # Do prod as only working with single dimensional outputs
    #y = prod(sin.(x))

    return x, y # e.g. sin(x) or for 2 variables sin(x+y) etc
end

function initialize_pivots_and_cache(func, min_grid, max_grid, d, N_total, num_starting_pivots, num_variables, N)

    """
    Initializes the pivot structures and function evaluation cache using `num_starting_pivots` random pivots.

    # Arguments
    - `func`: A function that takes a dinary vector and maps it to a grid value and corresponding function value.
    - `min_grid::Vector{Float64}`: Lower bounds of the input domain.
    - `max_grid::Vector{Float64}`: Upper bounds of the input domain.
    - `d::Int`: Local physical dimension (number of digit choices in the dinary system).
    - `N_total::Int`: Total number of sites (i.e., length of dinary vectors).
    - `num_starting_pivots::Int`: Number of initial random pivots to generate.
    - 'num_variables': an input for func
    - 'N' int for inputting to func

    # Returns
    - `row_pivots::Vector{Vector{Vector{Int}}}`: A list of lists containing partial row indices up to each site.
    - `col_pivots::Vector{Vector{Vector{Int}}}`: A list of lists containing partial column indices starting from each site.
    - `func_cache::Dict{Vector{Int}, Tuple{Any, ComplexF64}}`: A cache storing function evaluations for each unique dinary input.
    """

    # Step 1: Create a dictionary `func_cache` to store previously evaluated dinary vectors and their function values.
    func_cache = Dict{Vector{Int}, Tuple{Any, ComplexF64}}()

    # Step 2: Generate a list of `num_starting_pivots` random dinary vectors.
    # N_total random integers between 1 and d
    dinary_vectors = [rand(1:d, N_total ) for _ in 1:num_starting_pivots]

    # Step 3: Initialize the row and column pivot lists using the first random dinary.
    # - row_pivots: list where row_pivots[l+1] contains the prefix dinary[1:l] (empty when l=0).
    # - col_pivots: list where col_pivots[l+1] contains the suffix dinary[l+1:end] (empty when l=N_total).
    dinary = dinary_vectors[1]
    row_pivots = [[dinary[1:l]] for l in 0:N_total-1]  # list starts at 0 as 0th element is empty
    col_pivots = [[dinary[l+1:N_total]] for l in 1:N_total] # list starts at 1 and final element (N+1th) is empty

    # Step 4: Evaluate the function at the first dinary vector.
    x, y = func(dinary, min_grid, max_grid, d, num_variables, N)  # func outputs the grid location and the function value at that point
    func_cache[dinary] = (x, y)  # store both

    # Step 5: Loop through the remaining dinary vectors.
    # For each:
    #   - Append appropriate row and column slices to row_pivots and col_pivots. Don't want duplicates???
    #   - If the dinary is not already in the cache, evaluate it using `func` and store the result.

    for i in 2:num_starting_pivots  # do for all the dinarys except the first which has already been done

        dinary = dinary_vectors[i]
        if !(dinary in keys(func_cache))  # check if key already in dictionary before adding

            x , y = func(dinary, min_grid, max_grid, d, num_variables, N)
            func_cache[dinary] = (x, y)


            
            for l in 0:N_total-1                          
                if l != 0
                    #if !(dinary[1:l] in row_pivots[l+1])  
                        push!(row_pivots[l+1], dinary[1:l])  # dont do this for 0 because there is no row_pivots[0]
                    #end
                end

                if l != N_total-1
                    #if !(dinary[l+1:end] in col_pivots[l+1])
                        push!(col_pivots[l+1], dinary[l+2:N_total])  # dont do this for l=n-1 because there is no dinary[n+1]
                    #end
                end
            end
        end
    end


    return row_pivots, col_pivots, func_cache
end

function get_mps_from_pivots(row_pivots, col_pivots, min_grid, max_grid, d, N_total, func_cache, max_pivots, tolerance, func, num_variables, N)
    """
    Constructs a Matrix Product State (MPS) from pivot sets and a function cache.
    It also prunes small pivots from the random initialization by performing LU decompositions.

    # Arguments
    - `row_pivots::Vector{Vector{Vector{Int}}}`: List of row pivots for each site.
    - `col_pivots::Vector{Vector{Vector{Int}}}`: List of column pivots for each site.
    - `min_grid::Vector{Float64}`: Minimum grid bounds per variable.
    - `max_grid::Vector{Float64}`: Maximum grid bounds per variable.
    - `d::Int`: Local physical bond dimension.
    - `N_total::Int`: Total number of MPS sites.
    - `func_cache::Dict{Vector{Int}, Tuple{Any, ComplexF64}}`: Cache for storing function evaluations.
    - `max_pivots::Int`: Maximum rank of pivots used in truncation.
    - `tolerance::Float64`: Threshold for determining when to truncate.
    - `func`: Function used to compute evaluations at grid points.
    - 'num_variables': an input for func
    - 'N': int for inputting to func

    # Returns
    - `mps_list::Vector{Array{ComplexF64}}`: List of MPS tensors.
    - `row_pivots`: Updated row pivot sets.
    - `col_pivots`: Updated column pivot sets.
    """

    # Initialize lists to hold the T tensors and P inverses (from LU decompositions)
    T_tensors = []
    P_matrices = []

    # Loop over each site (1 to N_total - 1)
    for i in 1:N_total-1
            
        # Step 1a: Build the p matrix using func_cache or by evaluating func for each (row, col) pair.
        p = zeros(ComplexF64, length(row_pivots[i+1]), length(col_pivots[i]))  # length of row pivots and columns pivots should be the same
        
        # Extract rows and cols for the current site
        rows = row_pivots[i+1]  # equivilant of l as start counting at 1
        cols = col_pivots[i]  # l+1

        for (n, row) in pairs(rows)  # Loop over all the values in the pivot lists 
            for (m, col) in pairs(cols)
                v = vcat(row, col)
                if haskey(func_cache, v)
                    x, y = func_cache[v]  # if the dinary is in the dictionary then just retrieve the value
                    p[n,m] = y
                else  # if its not then must use func and add
                    x, y = func(v,  min_grid, max_grid, d, num_variables, N)
                    func_cache[v] = (x, y)
                    p[n,m] = y
                end
            end
        end

        # Step 1b: Perform LU decomposition with full pivoting and update the row and col pivot sets.
        # Use the provided lu_full_pivoting() function and extract new pivots and pivot_error.
        # Replace row_pivots[i+1] and col_pivots[i] with their updated versions.
    
        P, L_truncated, D_truncated, U_truncated, Q, new_row_pivots, new_col_pivots, pivot_error = lu_full_pivoting(p, max_pivots, tolerance, rows, cols)

        row_pivots[i+1] = new_row_pivots
        col_pivots[i] = new_col_pivots
    
    
        # Step 1c: Rebuild the p matrix using the new pivot sets and compute its inverse.
        # below same as 1a but with new pivots ___ 
        p = zeros(ComplexF64, length(row_pivots[i+1]), length(col_pivots[i]))  # length of row pivots and columns pivots should be the same
        
        # Extract rows and cols for the current site
        rows = row_pivots[i+1]  # equivilant of l as start counting at 1
        cols = col_pivots[i]  # l+1

        for (n, row) in pairs(rows)  # Loop over all the values in the pivot lists 
            for (m, col) in pairs(cols)
                v = vcat(row, col)
                if haskey(func_cache, v)
                    x, y = func_cache[v]  # if the dinary is in the dictionary then just retrieve the value
                    p[n,m] = y
                else  # if its not then must use func and add
                    x, y = func(v,  min_grid, max_grid, d, num_variables, N)
                    func_cache[v] = (x, y)
                    p[n,m] = y
                end
            end
        end
        
        #pretty_table(p)

        # Push the inverse into the list of p_tensors.
        inv_p = inv(p)  # find its inverse
        push!(P_matrices, inv_p)  # add it to list of tensors

    end


    # Step 2: Construct the T tensors for each site (1 to N_total)
    # Tensors have shape depending on their position in the MPS:
    #   - First site: (d, bond_dim)
    #   - Last site: (bond_dim, d)
    #   - Middle sites: (bond_dim_left, d, bond_dim_right)
    for i in 1:N_total

        # Extract rows and cols for the current site
        rows = row_pivots[i]  # equivilant of l-1 as start counting at 1
        cols = col_pivots[i]  # l+1

        # Build tensor based on site position:
        # For each valid dinary made by combining row, s, col (or just s and col, etc.),
        # retrieve function value (using cache if possible) and insert into tensor.
        
        # - For first site: no left bond
        if i == 1
            t = Array{ComplexF64}(undef, d, length(cols))
            for s in 1:d
                for (j, col) in pairs(cols)  # loop over each column and each possible value of s
                    v = vcat(s, col)
                    if haskey(func_cache, v)
                        x, y = func_cache[v]  # if the dinary is in the dictionary then just retrieve the value
                        t[s,j] = y
                    else  # if its not then must use func and add
                        x, y = func(v,  min_grid, max_grid, d, num_variables, N)
                        func_cache[v] = (x, y)
                        t[s,j] = y
                    end
                end
            end
        # - For last site: no right bond
        elseif i == N_total
            t = Array{ComplexF64}(undef, length(rows), d)
            for (j, row) in pairs(rows)
                for s in 1:d
                    v = vcat(row, s)  # make the vector itstring we are working with
                    if haskey(func_cache, v)
                        x, y = func_cache[v]  # if the dinary is in the dictionary then just retrieve the value
                        t[j, s] = y
                    else  # if its not then must use func and add
                        x, y = func(v,  min_grid, max_grid, d, num_variables, N)
                        func_cache[v] = (x, y)
                        t[j, s] = y
                    end
                end
            end
        # - Otherwise: full 3-index tensor
        else
            t = Array{ComplexF64}(undef, length(rows), d, length(cols))
            for (j, row) in pairs(rows)
                for s in 1:d
                    for (k, col) in pairs(cols)
                        v = vcat(row, s, col)
                        if haskey(func_cache, v)
                            x, y = func_cache[v]
                            t[j,s,k] = y
                        else 
                            x, y = func(v,  min_grid, max_grid, d, num_variables, N)
                            t[j,s,k] = y
                        end
                    end
                end
            end
        end  

        # Push each completed tensor to t_tensors
        push!(T_tensors, t)

    end


    # Step 3: Initialize mps_list with dummy tensors of correct shape for each site.
    # Use shape (d,1), (1,d), or (1,d,1) depending on position.
    # loop and push empty tensors to the list? dimensions are (d, legnth(row/col_pivots[i]))??
    
    
    # Step 4: Construct the final MPS tensors
    # - For i = 2 to N_total, contract p_tensors[i-1] with t_tensors[i] over matching indices
    mps_list = Vector{Array{ComplexF64}}()
    push!(mps_list, T_tensors[1])
    for i in 2:N_total  # Contract P and T for the rest of the tensors
        M = tensor_contraction(P_matrices[i-1], (2,), T_tensors[i], (1,))
        push!(mps_list, M)
    end

    return mps_list, row_pivots, col_pivots  # Do I not need to return errors??
end

function get_decomposition_and_updated_pivots(l, row_pivots, col_pivots, min_grid, max_grid, max_pivots, tolerance, func, func_cache, d, num_variables, N) # Need to test

    """
    Constructs the matrix Pi for a given level `l`, performs LDU decomposition, and retrieves updated pivot info.

    Inputs:
    l = level in the tensor contraction algorithm
    row_pivots, col_pivots = current list of multi-index row/col pivots
    min_grid, max_grid = domain boundaries for function evaluation
    max_pivots = truncation parameter
    tolerance = stopping tolerance for pivot selection
    func = function used to evaluate the matrix entries
    func_cache = dictionary to cache results of func

    Return:
    P, L, L11, L21, D, U, U11, U12, Q = decomposed components
    new_row_pivots, new_col_pivots = updated row and column pivots
    pivot_error = estimated decomposition error
    func_cache = updated cache dictionary
    """

    # Select the row pivots at level `l` and column pivots at level `l+1`
    row_pivots_l = row_pivots[l]
    col_pivots_l = col_pivots[l+1]

    # Initialize empty lists all_rows and all_cols to keep track of multi-indices for rows and columns
    all_rows = []
    all_cols = []

    # Create a zero-initialized 4D complex array Pi with dimensions: (Eq. 33c)
    Pi = zeros(ComplexF64, length(row_pivots_l), d, d, length(col_pivots_l))

    # Set a boolean flag to true to help fill all_cols only once
    flag = true
    
    # Loop over physical index row_s_idx from 1 to d:
    for row_s_idx in 1:d
        #   Loop over each row pivot (index and value) in row_pivots_l: 
        for (row_index, row) in enumerate(row_pivots_l)
            # Append the concatenation of the row pivot and row_s_idx to all_rows
            r = vcat(row, row_s_idx)
            push!(all_rows, r)
            # Loop over each column pivot (index and value) in col_pivots_l:
            for (col_index, col) in enumerate(col_pivots_l)
                # Loop over physical index col_s_idx from 1 to d:
                for col_s_idx in 1:d
                    # If flag is true, append the concatenation of col_s_idx and the column pivot to all_cols
                    if flag
                        c = vcat(col_s_idx, col)
                        push!(all_cols, c)
                    end
                    # Form a multi-index dinary by concatenating row pivot, row_s_idx, col_s_idx, and column pivot
                    dinary = vcat(row, row_s_idx, col_s_idx, col)
                    # If dinary is in func_cache, retrieve (x,y) from cache
                    if haskey(func_cache, dinary)
                        x, y = func_cache[dinary]
                    else
                        # Otherwise, call func with arguments (dinary, min_grid, max_grid, d, num_variables, N) to get (x,y), then cache it
                        x, y = func(dinary, min_grid, max_grid, d, num_variables, N)
                        func_cache[dinary] = (x, y)
                    end

                    Pi[row_index, row_s_idx, col_s_idx, col_index] = y
                end
            end

            flag = false # After finishing the inner loops for the first row pivot, set flag to false - all_cols has been filled and doesnt need to be again
        end
    end

    # Reshape Pi into a 2D matrix with dimensions:
    # (length(row_pivots_l) * d) rows and (d * length(col_pivots_l)) columns
    # (Note: Julia uses column-major order; the for-loop ordering in Step 5 ensures correct reshape)
    Pi = reshape(Pi, length(row_pivots_l) * d, d * length(col_pivots_l))

    # Call lu_full_pivoting on Pi with max_pivots, tolerance, all_rows, and all_cols
    P, L, D, U, Q, new_row_pivots, new_col_pivots, pivot_error = lu_full_pivoting(Pi, max_pivots, tolerance, all_rows, all_cols) 

    row_pivots[l+1] = new_row_pivots  # update pivots I_l and J_l+1
    col_pivots[l] = new_col_pivots

    # Extract blocks from L and U matrices for MPS canonical form assignment: (Eq. 27)
    L11 = UnitLowerTriangular(L[1:size(L, 2), :]) # Top square section of L with 1s along diagonal
    L21 = L[size(L, 2)+1:end, :]  # Bottom rectangular section
    U11 = UnitUpperTriangular(U[:, 1:size(U, 1)])  # First upper diagonal square section of U
    U12 = U[:, size(U,1)+1:end]  # Rest of U

    return P, L, L11, L21, D, U, U11, U12, Q, new_row_pivots, new_col_pivots, pivot_error, func_cache
end

function tci(N, func, min_grid, max_grid, tolerance, max_pivots, sweeps, d, func_cache, row_pivots, col_pivots, mps_list, num_variables)  # has issues

    """
    Performs the Tensor Cross Interpolation (TCI) algorithm to approximate a high-dimensional tensor 
    using matrix product states (MPS).

    Inputs:
    N = number of dimensions (sites)
    func = function to sample the tensor
    min_grid, max_grid = bounds of the domain
    tolerance = stopping criterion for pivot values
    max_pivots = maximum number of pivots allowed per decomposition
    sweeps = number of left-right sweeps for refinement

    Return:
    mps_list = list of MPS tensors approximating the original tensor
    pivot_error_list = per-site pivot errors during final sweep
    func_cache = dictionary of evaluated tensor entries
    """

    # Initialize a zero array pivot_error_list of length N-1 to store LDU decomposition errors for each link
    pivot_error_list = zeros(N-1)

    # Extract bond dimensions from mps_list tensors for sites 2 to N, storing them in bonddims
    bond_dims = [size(tensor, 1) for tensor in mps_list[2:end]]

    # Make a copy of bonddims into bonddims_previous to track changes between sweeps
    bond_dims_previous = bond_dims
    
    # Initialize final_sweep to 1 (to track how many sweeps were completed)
    final_sweep = 1

    # for saving frames to
    frames = String[]

    # Loop over sweep index from 1 to sweeps
    for sweep_index = 1:sweeps
        # Loop over direction flag dir in (true, false), where:
        #   - true means forward sweep (left to right)
        #   - false means backward sweep (right to left)
        for flag in (true, false)
            # Depending on dir, set range_l:
            #   - If dir == true, range_l = 1 to N-1 (forward)
            #   - Else, range_l = N-1 down to 1 (backward)
            if flag
                range_l = collect(1:N-1)
            else
                range_l = collect(N-1:-1:1)
            end

            # Loop over site index l in range_l
            for l in range_l

                P, L, L11, L21, D, U, U11, U12, Q, new_row_pivots, new_col_pivots, pivot_error, func_cache = get_decomposition_and_updated_pivots(l, row_pivots, col_pivots, min_grid, max_grid, max_pivots, tolerance, func, func_cache, d, num_variables, N)


                # Based on dir, compute left_tensor and right_tensor (Eq. 41 - *important to see my notes as well here since the paper is missing some detail on P and Q*)
                if flag  # Foward sweep - T_l*(inv(P_l)), T_l+1
                    left_tensor = transpose(P)*[ Array{ComplexF64}(I, size(L11, 1), size(L11, 1)); L21*inv(L11) ]   # ; is vertical concatenation
                    right_tensor = [L11*D*U11  L11*D*U12]*transpose(Q)
                else  # Bakcwards sweep - T_l, inv(P_l)*T_l+1
                    left_tensor = transpose(P)*[L11*D*U11; L21*D*U11]
                    identity = Array{ComplexF64}(I, size(U11, 1), size(U11, 1))
                    right_tensor = [ identity inv(U11)*U12 ]*transpose(Q)  # ' is horizontal concatenation 
                end  

                # Reshape left_tensor to its MPS form:
                if l != 1 # if not left boundary - reshape
                    left_tensor = reshape(left_tensor, div(size(left_tensor, 1), d), d, size(left_tensor, 2))  
                end 
                
                # Reshape right_tensor to its MPS form:
                if l != N-1 # if not right boundary - reshape
                    right_tensor = reshape(right_tensor, size(right_tensor, 1), d, div(size(right_tensor, 2), d))  # Dividing one side by physical dimension size d
                end  

                # Update mps_list at positions l and l+1 with left_tensor and right_tensor respectively
                mps_list[l] = left_tensor  
                mps_list[l+1] = right_tensor 
    
                # Update pivots I_l and J_l+1
                row_pivots[l+1] = new_row_pivots
                col_pivots[l] = new_col_pivots
    
                # Store pivot_error in pivot_error_list at position l
                pivot_error_list[l] = pivot_error
    
                # Update bond_dims[l] to bond dimension on the left of site l+1
                bond_dims[l] = size(right_tensor, 1)
    
            # End site loop
            end

        # End direction loop
        end

        # Check stopping condition:
        # If bonddims equals bonddims_previous AND sweep > 10:
        if bond_dims == bond_dims_previous && sweep_index > 10
            # Print a message about bond dimension convergence and return mps_list, pivot_error_list, func_cache, sweep, bonddims
            println("TCI has converged after $sweep_index sweeps with bond dimensions $bond_dims and pivot error $pivot_error_list")

            return  mps_list, pivot_error_list, func_cache, sweep_index, bond_dims, frames  # TCI has converged

        # Else copy bonddims to bonddims_previous for next sweep
        else
            bond_dims_previous = bond_dims
        end
    
        # Update final_sweep to current sweep
        final_sweep = sweep_index
    # End sweep loop
    end
    
    # If sweeps exhausted without convergence, print a message and return mps_list, pivot_error_list, func_cache, final_sweep, bonddims
    println("TCI has not converged after $final_sweep sweeps")

    return mps_list, pivot_error_list, func_cache, final_sweep, bond_dims, frames  # TCI has not converged
end

function mps_list_contraction(mps_list, N, d, N_total)
    """
    Takes an mps stored in a list and contracts it into a single vector of function values that can be plotted
    
    Inputs:
    mps_list - list of tensors in the mps
    N - number of sites in the mps
    d - physical dimensions of each site

    Outputs:
    C_vec - vector with all the values of the functions at each string in order ready to be plotted

    """

    Cont = tensor_contraction(mps_list[1], (2,), mps_list[2], (1,))
    for i in 3:length(mps_list)
        Cont = tensor_contraction(Cont, (length(size(Cont)),), mps_list[i], (1,))
    end 

    # Now have 3 dimesional tensor which needs to be converted to 1D list
    C_vec = []
    for i in 1: d^N_total
        dinary = decimal_to_dinary(i-1, N, d)  # i-1 as first value is 0
        push!(C_vec, Cont[dinary...])
    end

    return C_vec
end

function mps_list_to_itensors_mps(N, mps_list, sites)

    """
    Converts a list of MPS tensors (`mps_list`) into an ITensors.jl MPS object.

    # Inputs
    - `N::Int`: Number of sites in the MPS.
    - `mps_list::Vector{<:AbstractArray}`: List of tensors representing the MPS, where each tensor corresponds to a site and has bond and physical indices.
    - `sites::Vector{Index}`: Vector of physical indices for each site, compatible with ITensors.jl.

    # Returns
    - `mps::MPS`: The corresponding MPS object constructed from `mps_list` and `sites`, using ITensors.jl data structures.

    # Description
    This function constructs link indices representing the virtual bonds between sites, then builds ITensor objects for each site tensor with appropriate physical and link indices, and finally combines them into an ITensors MPS object.
    """

    # Create a list `links` of Index objects for the bond dimensions between sites 1 and N.
    links = Vector{Index}(undef, N-1)
    for n in 2:N
        # Create an Index with dimension = size of mps_list[n] along dimension 1
        dims = size(mps_list[n], 1)
        links[n-1] = Index(dims, "Link, i = $(n-1)")
        # Name the Index as "Link,i=$(n-1)" to indicate its position
    end

    # Initialize a vector `mps_tensors` to hold ITensor objects of length N
    mps_tensors = Vector{ITensor}(undef, N)

    # Construct the first ITensor `mps_tensors[1]` using:
    #   - the tensor mps_list[1]
    #   - the physical index sites[1]
    #   - the link index links[1]
    mps_tensors[1] = ITensor(mps_list[1], sites[1], links[1])

    # For sites n = 2 to N-1, construct intermediate ITensors:
    for n in 2:N-1
        # Use mps_list[n]
        # Use the left link index links[n-1]
        # Use the physical index sites[n]
        # Use the right link index links[n]
        mps_tensors[n] = ITensor(mps_list[n], links[n-1], sites[n], links[n])
    end

    # Construct the last ITensor `mps_tensors[N]`:
    #   - Use mps_list[N]
    #   - Use the left link index links[N-1]
    #   - Use the physical index sites[N]
    mps_tensors[N] = ITensor(mps_list[N], links[N-1], sites[N])

    # Use the ITensors constructor MPS(mps_tensors) to form an MPS object
    mps = MPS(mps_tensors)

    # Return the constructed MPS
    return mps
end

function integrate(mps, sites, d, N, N_total, min_grid, max_grid)

    """
    Approximates the integral of the tensor represented by the MPS `mps` over its domain using simple quadrature weights.

    # Inputs
    - `mps::MPS`: The Matrix Product State representing the tensor to be integrated.
    - `sites::Vector{Index}`: Physical indices corresponding to each site in the MPS.
    - `d::Int`: Physical dimension at each site.
    - `N::Int`: Number of active sites (dimensions).
    - `N_total::Int`: Total number of sites in the MPS.
    - `min_grid::Vector{Float64}`: Minimum bounds of the integration domain per dimension.
    - `max_grid::Vector{Float64}`: Maximum bounds of the integration domain per dimension.

    # Returns
    - `res::Number`: The approximate integral of the tensor represented by the MPS over the domain.

    # Description
    This function constructs an MPS representing uniform integration weights (all ones) over the physical dimensions,
    then contracts it with the input MPS to compute the integral approximation. The integration step size is computed
    assuming uniform discretization over the domain.

    """

    # Initialize an MPS `mps_int` using the physical `sites` indices.
    mps_int = MPS(sites)
    
    # Create link indices representing the bonds between sites of dimension 1.
    # For each bond from 1 to (N_total - 1), create an Index of size 1 and name it "Link,i=$n".
    links = Vector{Index}(undef, N_total-1)
    for n in 1:N_total-1
        links[n] = Index(1, "Link, i = $n")
    end

    # Define a vector `vec` of length `d` filled with ones to represent uniform integration weights.
    vec = ones(d)
    
    # Compute the integration step size `delta_x`
    delta_x = (max_grid .- min_grid)./(d^N-1)  # Just as in func
    
    # Loop over all sites i in 1 to N_total to construct ITensors for `mps_int`:
    for i in 1:N_total
        
        s = sites[i]

        #   - For the first site (i == 1), create an ITensor with shape (d, 1) using `vec`,
        #     attach physical index `sites[i]` and right link `links[i]`.
        if i == 1
            r = links[1]
            A = ITensor(s , r)
            for j in 1:d
                A[s=> j, r => 1] = vec[j]
            end
        #   - For the last site (i == N_total), create an ITensor with shape (d, 1),
        #     attach physical index `sites[i]` and left link `links[i-1]`.
        elseif i == N_total
            l = links[i-1]
            A = ITensor(l, s)
            for j in 1:d
                A[s => j, l => 1] = vec[j]
            end
        #   - For intermediate sites, create an ITensor with shape (d, 1, 1),
        #     attach physical index `sites[i]` and both left and right links.
        else
            l = links[i - 1]
            r = links[i]
            A = ITensor(l, s, r)
            for j in 1:d
                A[l => 1, s => j, r => 1] = vec[j]
            end
        end

        mps_int[i] = A  
    
    end

    # Compute the inner product of `mps_int` and `mps` using the `inner` function.
    inner = ITensors.inner(mps_int, mps)
    # Implement with contraction function also

    # Multiply the resulting scalar by `delta_x` to scale by the integration step size.
    res = inner*delta_x[1]  # here take [1] as delta_x can be a vector but should be same in all dimensions
    
    # Return the final integration result.
    return res
end

function qft(mps, sites, N)  # not working
    """
    Takes an MPS representation of a function and creates the quantum fourier transform MPO to apply to it 
    Returns the MPS of the fourier transform
    """

    # Can just create an MPO with ITensors
    os = OpSum()

    for k in 1:N
        for n in k:N  
            if n == k
                add!(os, 1 , "H", n)  # add a H gate at the top of the column of gates 
            else
                angle = pi/(2^(n-k))
                add!(os, 1, "Phase", n ; theta=angle)  # add the phase gate
            end
        end
    end

    QFT = MPO(os, sites)

    """
    or can build it without ITensors - hard
        build H tensor and phase tensors
        contstruct quantum circuit 
        svd up and down to contract into an MPO
    """

    F_MPS = contract(MPS, QFT)

    sites_rev = sites[end:-1:1]  # reverse order of the sites
    F_MPS = replaceinds(F_MPS, sites, sites_rev)  # relabel the MPS to use reversed sites

    return F_MPS 
end

# Initialise Pivots and MPS:
row_pivots, col_pivots, func_cache = initialize_pivots_and_cache(func, min_grid, max_grid, d, N_total, num_starting_pivots, num_variables, N)
mps_list, row_pivots, col_pivots = get_mps_from_pivots(row_pivots, col_pivots, min_grid, max_grid, d, N_total, func_cache, max_pivots, tolerance, func, num_variables, N)

# Perform TCI and convert to ITensors object:
mps_list, pivot_error_list, func_cache, final_sweep, bonddims, frames = tci(N_total, func, min_grid, max_grid, tolerance, max_pivots, sweeps, d, func_cache, row_pivots, col_pivots, mps_list, num_variables)
mps = mps_list_to_itensors_mps(N_total, mps_list, sites)

# Quantum Fourier Transform the function
#FT_mps = qft(mps, sites, N)
#plot(FT_mps)

# Contraction of MPS for plotting:
C_vec = mps_list_contraction(mps_list, N, d, N_total)

# Plotting:
x = range(min_grid[1], max_grid[1], d^N)
y = (sin.(5x) + 0.5cos.(3x.^2)).*exp.(-0.1x.^2) + atan.(x)/5 + 0.1.*x.*cos.(x)
#y = sin.(x)
p = plot(x, C_vec, linestyle = :dash, label="MPS approximation", xlabel="x", ylabel="f(x)", title = "N = 8, d = 2, Tolerence = 1e-14, Max Pivots = 100")
plot!(x, y, linewidth = 1.5, label = "Actual value")
# function cache
x_func = fixed_dinary_to_decimal.(keys(func_cache), d)  # Apply to all keys
y_func = [v[2] for v in values(func_cache)]  # Just take the value not the position in grid
sorted_indices = sortperm(x_func)
x_sorted = (max_grid[1]-min_grid[1]).*x_func[sorted_indices]./(d^N) .+ min_grid[1]
y_sorted = real.(y_func[sorted_indices])
scatter!(x_sorted, y_sorted, markersize = 3, ma = 0.3,mc = :red, label = "Function Evaluations")
display(p)
# label plot better 
#savefig("TCI function eval.pdf")


"""
# Analysis:
# want to know how many times the function was evaluated compared to total domain d^N
println("Percentage of domain evaluated: (length(func_cache)*100/(d^N))%")
# Want to know the compression percentage - number of points that are stored in our mps compared to full version with d^N points
points = [prod(size(mps_list[n])) for n in 1:length(mps_list)]
total = sum(points)
compression = total*100/(d^N)
println("Compression Percentage = (compression)%")
# Want to know (in)fidelity
norm_C = C_vec./norm(C_vec)
norm_y = y./norm(y)
fidelity = dot(norm_C, norm_y)
infidelity = abs(1-fidelity)^2
println("Infidelity: (infidelity)")

# Integrate:
res = integrate(mps, sites, d, N, N_total, min_grid, max_grid)
println(res)
"""

"""
# want to plot infidelity and compression against N and d
infidelity_vec = []
compression_vec = []
domain_vec =[]
d_vec = collect(2:6)
for d in 2:6

    # Initialise Pivots and MPS:
    row_pivots, col_pivots, func_cache = initialize_pivots_and_cache(func, min_grid, max_grid, d, N_total, num_starting_pivots, num_variables, N)
    mps_list, row_pivots, col_pivots = get_mps_from_pivots(row_pivots, col_pivots, min_grid, max_grid, d, N_total, func_cache, max_pivots, tolerance, func, num_variables, N)

    # Perform TCI and convert to ITensors object:
    mps_list, pivot_error_list, func_cache, final_sweep, bonddims = tci(N_total, func, min_grid, max_grid, tolerance, max_pivots, sweeps, d, func_cache, row_pivots, col_pivots, mps_list, num_variables)

    # Contraction of MPS for plotting:
    C_vec = mps_list_contraction(mps_list, N, d, N_total)

    x = range(min_grid[1], max_grid[1], d^N)
    y = (sin.(5x) + 0.5cos.(3x.^2)).*exp.(-0.1x.^2) + atan.(x)/5 + 0.1.*x.*cos.(x)

    # Calculate infidelity
    norm_C = C_vec./norm(C_vec)
    norm_y = y./norm(y)
    fidelity = dot(norm_C, norm_y)
    infidelity = abs(1-fidelity)^2

    # Want to know the compression percentage - number of points that are stored in our mps compared to full version with d^N points
    points = [prod(size(mps_list[n])) for n in 1:length(mps_list)]
    total = sum(points)
    compression = total*100/(d^N)

    # want to know how many times the function was evaluated compared to total domain d^N
    domain_eval = (length(func_cache)*100/(d^N))

    push!(infidelity_vec, infidelity)
    push!(compression_vec, compression)
    push!(domain_vec, domain_eval)
end

plot(d_vec, domain_vec,  xlabel="Physical dimension of sites", ylabel="Domain Evaluations", title="Percentage of domain evaluated by TCI output with N=8")
savefig("TCI domain evals vs d graph.pdf")

plot(d_vec, compression_vec,  xlabel="Physical dimension of sites", ylabel="Compression Percentage", title="Compression of TCI output with N=8")
savefig("TCI compression vs d graph.pdf")

plot(d_vec, infidelity_vec,  xlabel="Physical dimension of sites", ylabel="Infidelity", title="Infidelity of TCI output with N=8")
savefig("TCI infidelity vs d graph.pdf")
"""

println("\n Finished \n")