using ITensors, ITensorMPS
using KrylovKit
using LinearAlgebra
import ITensors.svd as ITensors_SVD
using Plots, LaTeXStrings

# DMRG Variables:
# Number of sites, sweeps of DMRG, cut-off error, max dim of mps
N = 64
max_sweeps = 10 
cut_off = 1e-10
max_dim = 10

# Schwinger Wilson Variables:
l0 = 0.125
m_lat = 0

function DMRG(N, max_sweeps, cut_off, max_dim, H_MPO, sites)
    """
    Takes in a Hamiltonian in the form of an MPO, creates random MPS to then variationally optimize two sites at a time
    Need to put the mps into canoncial form, find minimum energy eigenvector for two sites and update, then move to next
    Continues until max number of sweeps or energy change falls below threshold
    
    Outputs the energy, and the MPS state with this energy (ground state)
    """

    # Initialize a random MPS and put it in normalized canonical form using randomMPS, orthogonalize!
    psi = randomMPS(sites, max_dim)
    orthogonalize!(psi, 1)  # Do I want to put in LCF to start as first sweep is left to right?


    # Initialize an array H_eff_parts which will hold the left and right parts of the effective Hamiltonian
    # Here you can find prime and dag useful, each element should be the A_conj * mpo * A of a position from 1 to N where A is the mps ansatz tensor 
    
    left_envs = [ITensor(1)]
    right_envs = [ITensor(1)]  # right_envs[N+1] after last site

    for i in 1:N
        # Contract left_envs[i] with psi[i] (conj) * mpo[i] * psi[i]
        A = psi[i]
        A_conj = dag(prime(A))   # prime bra's link indices so different to ket's indicies
        # Contract Adag * mpo[i] * A
        tmp = A_conj * H_MPO[i] * A
        push!(left_envs, left_envs[i] * tmp)
    end

    for i in N:-1:1
        A = psi[i]
        A_conj = dag(prime(A))
        tmp = A_conj * H_MPO[i] * A
        pushfirst!(right_envs, tmp * right_envs[1])
    end


    # Initialize to zero E_curr which will change after a full sweep and E which will change within the sweep
    E_curr = 0.0
    E = 0.0

    # Perform the DMRG sweeps
    for sweep in 1:max_sweeps

        # Forward part of sweep from left to right
        for i in 1:N-1  # This is the left index of the two sites being updated so careful of the upper limit of the for loop
    
            # Get the effective Hamiltonian using the 2 sites from the mpo and the product of the rest of the positions from H_eff_parts, the prod function is useful, careful of the 3 cases (2 edge cases and 1 bulk)
            H_eff = left_envs[i] * H_MPO[i] * H_MPO[i+1] * right_envs[i+2]
            # Get the two site part of the mps mps_two_site
            mps_two_site = psi[i] * psi[i+1]
            

            # Now we call the eigensolver with H_eff and mps_two_site using eigsolve from KrylovKit and we want to specify :SR (smallest real) and that our H_eff is hermitian, it accepts an ITensors for H_eff and mps_two_site 
            # which are the case A::AbstractMatrix (https://jutho.github.io/KrylovKit.jl/stable/man/eig/#KrylovKit.eigsolve) so no reshaping etc is required
            # since H_eff and mps_two_site share indices, it knows how to reshape the H_eff by itself into a matrix and go from there, then return the appropriate indices for the eigenvectors
            vals, vecs = KrylovKit.eigsolve(H_eff, mps_two_site, 1, :SR; ishermitian=true)

            # Update current energy with the lowest we found from optimizing sites i, i+1
            E_curr = real(vals[1])
            # Define the lowest-eigenvalue eigenvector from the results of eigsolve
            M_new = vecs[1]

            # Perform using ITensors_SVD (to distinguish from svd in LinearAlgebra) the SVD decomposition on the lowest-eigenvalue eigenvector to get two mps tensors, you can get specific indices of a tensor using inds(t, :tags => "label_you_to_specify")
            # careful of the case of the first site vs all other, in the ITensors_SVD function we want to specify maxdim and cutoff as well as lefttags = "my_label" and righttags for the label of the column index of U and the row index of V in the SVD decomposition M = USV
            # see the documentation here https://docs.itensor.org/ITensors/stable/ITensorType.html#LinearAlgebra.svd-Tuple%7BITensor,%20Vararg%7BAny%7D%7D, the S we contract with V (why not with U?)
            U, S, V = ITensors_SVD(M_new,commoninds(M_new, psi[i]); maxdim=max_dim, cutoff=cut_off, lefttags = "Link, l=$(i)", righttags = "Link, l=$(i+1)")
            
            # Update the mps 
            psi[i] = U
            psi[i+1] = S * V

            # Update the H_eff_parts at positions i and i+1 - only need to update left side as sweeping left to right
            Adag = dag(prime(psi[i]))
            tmp = Adag * H_MPO[i] * psi[i]
            #tmp = noprime(tmp)  # remove primes
            left_envs[i+1] = left_envs[i] * tmp

            # Here you can print the sweep number, the position of the mps just optimized, the norm of the mps (using the norm function), and the maximum bond dimension of the mps using the maxlinkdim function as well as the energy E
            @info "Sweep $sweep L->R: Optimised sites ($i,$(i+1)); E = $(E_curr); norm(psi) = $(norm(psi)); maxlinkdim = $(maxlinkdim(psi))"
        end

        # Backward part of sweep from right to left
        for i in N:-1:2 # This is the left index of the two sites being updated
    
            # Get the effective Hamiltonian
            H_eff = left_envs[i-1] * H_MPO[i-1] * H_MPO[i] * right_envs[i+1]
            
            # Get the two site part of the mps
            mps_two_site = psi[i-1] * psi[i]

            # Now we call the eigensolver with H_eff and mps_two_site
            vals, vecs = KrylovKit.eigsolve(H_eff, mps_two_site, 1, :SR; ishermitian=true)

            # Update current energy with the lowest we found from optimizing sites i, i+1
            E_curr = real(vals[1])

            # Define the lowest-eigenvalue eigenvector
            M_new = vecs[1]

            # Perform the SVD decomposition on it to get two mps tensors
            U, S, V = ITensors_SVD(M_new, commoninds(M_new, psi[i-1]); maxdim=max_dim, cutoff=cut_off, lefttags = "Link, l=$(i-1)", righttags = "Link, l=$(i)")

            # Update the mps
            psi[i-1] = U * S
            psi[i] = V

            # Update the H_eff_parts - here need to update right_envs[i] and [i-1]
            Adag2 = dag(prime(psi[i]))
            tmp2 = Adag2 * H_MPO[i] * psi[i]
            # tmp2 = noprime(tmp2)  # remove primes
            right_envs[i] = tmp2 * right_envs[i+1]
            # Update right_envs[i-1] because psi[i-1] changed

            if i == 2 # Then update the i=1 site in case we end the loop here
                Adag = dag(prime(psi[i-1]))
                tmp = Adag * H_MPO[i-1] * psi[i-1]
                # tmp = noprime(tmp)  # remove primes
                right_envs[i-1] = tmp * right_envs[i]
            end

            # Print the sweep number, the position of the mps just optimized, the norm of the mps, and the maximum bond dimension of the mps using the maxlinkdim function as well as the energy E
            @info "Sweep $sweep R->L: Optimised sites ($(i-1),$(i)); E = $(E_curr); norm(psi) = $(norm(psi)); maxlinkdim = $(maxlinkdim(psi))"
        end

        # Compute the fractional energy change after a full sweep absolute value of (E - E_curr) / E_curr
        frac = abs((E-E_curr)/(E+0.00000001))  # Ensure not dividing by 0

        # Set the current energy variable E_curr to the final decreased energy E after a full sweep
        E = E_curr
        
        # Check for stopping conditions, if fractional energy is smaller than a given input cut_off or if we reached the maximum number of sweeps we break the sweeping loop
        if frac < cut_off
            @info "Converged after $sweep sweeps with fractional change < $cut_off."
            break
        end

    end

    # Return E_curr for the ground state energy
    return E, psi

end

function FirstExcited(H_MPO, psi, lambda, sites)
    """
    Takes a Hamiltonian MPO and its ground state
    Produces a new Hamiltonian H → H' = H + λ |psi⟩ ⟨psi| (eq 2.28 in Takis Thesis)
    New Hamiltonians ground state is 1st excited of input Hamiltonian if lambda > energy gap

    It is more efficient to enforce that the solution being found by DMRG is orthogonal to the GS when performing the optimisation
    Using orthgonality, only the site that is being updated needs to be updated orthognally to the corresponding site in the groundstate if also in the same canoncial form
    """
    
    P = projector(psi)  # Constructs the projector of the first ground state
    
    # Having issues with the sites of P and H as they are different labels and so it wont let me sum them

    H = H_MPO + lambda * P  # Projects out ground state

    return H
end

# Hamiltonian constructions:
function Heisenburg(N)
    """
    Creates an MPO for the Heinsenberg Hamiltonian on N sites
    H_j = (Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + Sz_j Sz_{j+1})
    Output: ITensors MPO
    """

    # Define sites object using siteinds
    sites = ITensors.siteinds("S=1/2", N)

    # Define the Hamiltonian MPO with OpSum and MPO constructor:
    os = OpSum()

    # Heinsenberg Model Hamiltonian: H_j = (Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + Sz_j Sz_{j+1})
    for j in 1:N-1
        add!(os, 0.5, "S+", j, "S-", j+1)   # S⁺_j S⁻_{j+1}
        add!(os, 0.5, "S-", j, "S+", j+1)   # S⁻_j S⁺_{j+1}
        add!(os, 1.0, "Sz", j, "Sz", j+1)   # Sᶻ_j Sᶻ_{j+1}
    end

    H_MPO = MPO(os, sites)
    return H_MPO, sites
end

function Ising(N, J, gz, gx)
    """
    Constructs an MPO of the Ising model Hamiltonian on N sites for a given J, gx, gz
    H = -J sum(Sz_j Sz_j+1) - gx sum(Sx_j) - gz sum(Sz_j)
    Input: 
        J nearest neighbor coupling constant
        gx Transverse field strength
        gz Longitudinal field strength - small (0.1) to make ground state non-degenerate
    Output: 
        ITensors MPO
    """
        
    # Define sites object using siteinds
    sites = ITensors.siteinds("S=1/2", N)

    # Define the Hamiltonian MPO with OpSum and MPO constructor:
    os = OpSum()
    # Transverse Field Ising Model Hamiltonian: H = -J sum(Sz_j Sz_j+1) - gx sum(Sx_j) - gz sum(Sz_j)

    # Nearest-neighbor ZZ term
    for j in 1:(N-1)
        add!(os, J, "Sz", j, "Sz", j+1)
    end
    # Transverse field gx * X_i
    for j in 1:N
        add!(os, gx, "Sx", j)
    end
    # Longitudinal field gz * Z_i
    for j in 1:N
        add!(os, gz, "Sz", j)
    end

    H_MPO = MPO(os, sites)
    return H_MPO, sites
end

function AAHarper(N, mu, V)
    """
    Constructs MPO for Aubry-Andre-Harper Model: H = (1/4) Σ J (S⁺_j S⁻_{j+1} + S⁻_j S⁺_{j+1}) + Σ h_j Sz_j
    Inputs:
        N number of sites
        mu offdiagonal amplitude
        V on site amplitude
    """

    delta = 0  # can vary from -pi -> pi
    a = (sqrt(5)-1)/2  # irrational frequency

    J = [(1+ mu * cos(2*pi*(j+0.5)*a + delta)) for j in 1:N]
    h = [V*cos(2*pi*j*a + delta) for j in 1:N] 

    for j in 1:N-1  # Hopping terms
        add!(os, J[j], "S+", j, "S-", j+1)
        add!(os, J[j], "S-", j, "S+", j+1)
    end
    for j in 1:N  # On-site terms
        add!(os, h[j], "S+", j, "S-", j)
    end

    H_MPO = MPO(os, sites)

    return H_MPO, sites
end

function Schwinger_Wilson(N, l0, m_lat, x)
    """
    Creates Wilson Hamiltonian MPO for the Schwinger model as found in https://doi.org/10.1103/PhysRevD.108.014516
    Inputs
        N - Number of sites
        theta - Topological term between 0-2pi
        g - Coupling constant
        m_lat - lattice mass
        
    """
    # Define sites object
    sites = ITensors.siteinds("S=1/2", 2*N)  # need 2N sites for N physical

    # Define the Hamiltonian MPO with OpSum and MPO constructor:
    os = OpSum()

    # Variables
    r = 1  # Wilson Parameter, usually 1 
    g = 1
    # a =  1 # Lattice Spacing 
    # x = 1/(a*g)^2
    # l0 = theta/(2*pi)
    lambda = 100 # Lagrange multiplier that must be 'sufficiently large' - ensures no ground state degeneracy
    
    # eq 1.27 
    for n in 0:(N-2)
        # add!(os, x*(r-1) , "S+", 2n+1, "Z", 2n+2, "Z", 2n+3, "S-", 2n+4)  (r-1 will be 0)
        add!(os, 0.5*x*(r+1) , "X", 2n+2, "X", 2n+3)
        add!(os, 0.5*x*(r+1) , "Y", 2n+2, "Y", 2n+3)
    end
    
    coeff = (m_lat/g)*sqrt(x) + x*r
    for n in 0:N-1
        add!(os, coeff , "X", 2n+1, "X", 2n+2)
        add!(os, coeff , "Y", 2n+1, "Y", 2n+2)
    end
    
    for n in 0:(2N-1)
        for k in (n+1):(2N-1)
            add!(os, 0.5*(N-ceil((k+1)/2)+lambda) , "Z", n+1, "Z", k+1)
        end
    end
    
    for n in 0:(2N-3)
        add!(os, l0*(N-ceil((n+1)/2)) , "Z", n+1)
    end
    
    # Constant Energy Shift
    E0 = (l0^2)*(N-1)+ 0.25*N*(N-1) + lambda*N/2
    add!(os, E0, "Id", 1)
    
    H_MPO = MPO(os, sites)
    return H_MPO, sites
end

# Observables:
function L_W(N, l0, sites)
    """
    Creates the Electric field density operator for Wilson Fermions
    L_W = l0 + 0.5 sum{0->N/2-1}(Z_2k + Z_2k+1)
    """

    # Define the Hamiltonian MPO with OpSum and MPO constructor:
    os = OpSum()

    for n in 0:(ceil(div(N,2))-1)
        add!(os, 0.5 , "Z", 2*n+1)
        add!(os, 0.5 , "Z", 2*n+2)
    end

    # Constant Shift
    shift = l0
    add!(os, shift, "Id", 1)

    L_MPO = MPO(os, sites)

    return L_MPO 
end

function P_W(N, sites)
    """
    Creates the Particle Number operator for Wilson Fermions
    P_W = N + 0.5 sum{0->N-1}(X_2k X_2k+1 + Y_2k Y_2k+1)
    """

    # Define the Hamiltonian MPO with OpSum and MPO constructor:
    os = OpSum()

    for n in 0:(N-1)
        add!(os, 0.5 , "X", 2n+1, "X", 2n+2)
        add!(os, 0.5 , "Y", 2n+1, "Y", 2n+2)
    end

    # Constant Energy Shift
    shift_per_site = N
    add!(os, shift_per_site, "Id", 1)


    P_MPO = MPO(os, sites)

    return P_MPO
end

function Q_W(N, sites)
    """
    Creates the Charge operator for Wilson Fermions
    Q_n =  (Z_n + (-1)^n)/2
    """

    # Define the Hamiltonian MPO with OpSum and MPO constructor:
    os = OpSum()

    for n in 0:(N-1)
        add!(os, 0.5 , "Sz", 2*n+1)
        add!(os, 0.5 , "Sz", 2*n+2)
    end

    Q_MPO = MPO(os, sites)

    return Q_MPO
end

function Magnetisation(psi)
    # Magnetisation per site:
    magz = expect(psi, "Sz")
    for (j,mz) in enumerate(magz)
        #println("j mz")
    end
    total_M = sum(magz)/N
    # println("Magnetisation per site: ", total_M/N)  # Magnetisation per site
    return total_M
end

function Entanglement_Entropy(psi)
    """
    Finds the entanglement entropy of an mps
    Evaluates it at the middle of the mps - EE should be roughly const but drop at ends
    Using the middle reduces boundary effects
    """
    b = 8
    psi_N = orthogonalize!(psi, b)
    U,S,V = svd(psi_N[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
    SvN = 0.0
    for n=1:dim(S, 1)
        p = S[n,n]^2
        SvN -= p * log(p)
    end
    return SvN
end

# Schwinger Model Analysis:
function Critical_point_convergence(l0_min = 1.2, l0_max = 1.45, cutoff_min = 10, cutoff_max = 16)
    """
    Uses DMRG to find the ground state of the Schwinger Model with Wilson Fermions
    Varies the cutoff value of DMRG for different N to see how this affects the location of the phase transition
    Plots convergence of the location l0* for each N as cutoff varies
    """
    
    p = plot(title="Phase transition location for different cutoffs, \n maxdims=10", xlabel="cutoff value, 1e-x", ylabel="Phase Transition Location, l0")
    
    lambda = 100
    m_lat = 10  # > 0.33 where we see the critical point
    max_dim = 10
    max_sweeps = 10

    # plot how the phase transition location varies for different cutoff values or maxdims at different N
    #cut_off_list = range(1e-7, 1e-14, 8)

    l0_list = range(l0_min, l0_max, 15)
    for N in 4:7

        x = (N/30)^2  # fix N/sqrt(x) = 30, volume of the system
        l_list = []

        for i in cutoff_min:cutoff_max
            #exponent = cut_off_list[i]
            cut_off = 10.0^(-i)
            for j in 1:10
                l0 = l0_list[j]

                H_MPO, sites =  Schwinger_Wilson(N, l0, m_lat, x)
                # Call the dmrg function
                E, psi = DMRG(2*N, max_sweeps, cut_off, max_dim, H_MPO, sites)  
                println("\n")

                # Calculate expection of particle number - should be 0 for low l and then 2
                P = P_W(N, sites)
                ex_P = real(inner(psi',P,psi))
                
                if ex_P > 1
                    push!(l_list, l0)
                    break
                elseif j == 10  # to make sure everything is the same length
                    push!(l_list, l0)
                end
            end
        end
        plot!(p, range(10, 16, 7), l_list, label = "N=$N")
    end

    return p
end
#p = Critical_point_convergence(1.2, 1.45, 8, 14)
#display(p)
#savefig(p, "Critical point variation with cutoff 3.pdf")

function Schwinger_1storder_PhaseTransition(N, l0_min , l0_max, points)

    """
    Uses DMRG to find ground state of Schwinger Model with Wilson Fermions
    Produces a graph of the 1st order phase transition of the Schwinger model
    for high m/g we see a phase transition where particle number goes from 0-2 and field strength drops 
    varies l0 and finds expectation values of PN and EFS, returning the plot
    """

    particles = []
    field = []
    l0_list = range(l0_min, l0_max, points)

    x = (N/30)^2  # fix N/sqrt(x) = 30, volume of the system
    lambda = 100 # large?
    m_lat = 10  # > 0.33 where we see the critical point

    p = plot(title="Expectation values for Wilson Fermions, \n N=$N, N/sqrt(x)=30, m_lat=$m_lat", xlabel="Background Field Strength, θ/2π")
    
    for i in 1:points # Vary l0, background field

        # Schwinger Model Variables:
        l0 = l0_list[i]

        # Create Hamiltonian MPO from one of the functions above ^^
        H_MPO, sites =  Schwinger_Wilson(N, l0, m_lat, x)

        # Call the dmrg function
        E, psi = DMRG(2*N, max_sweeps, cut_off, max_dim, H_MPO, sites)  
        println("\n")

        # Calculate expection of particle number - should be 0 for low l and then 2
        P = P_W(N, sites)
        ex_P = real(inner(psi',P,psi))
        println("Particle number: ", ex_P)
        push!(particles, ex_P)

        # Calculate expection of EFS 
        L = L_W(N, l0, sites)
        ex_L = real(inner(psi',L,psi))
        println("Electric field strength: ", ex_L)
        push!(field, ex_L)

        # Calculate total charge - should be 0 for the state to be physical 
        # can also calculate for each site to see where charge is distributed
        Q = Q_W(N, sites)
        ex_Q = real(inner(psi', Q, psi))
        println("Total charge: ", ex_Q)
    
    end

    plot!(p, l0_list, particles, label = "Particle number", )
    plot!(p, l0_list, field, label = "Field Strength")
    
    return p
end
#p = Schwinger_1storder_PhaseTransition(6, 1 , 2, 5)
#display(p)

#savefig("Schwinger 1st order phase transition.pdf")
#filename = "Schwinger 1st order phase transition.pdf"
#println("Saved plot to: ", abspath(filename))

function Schwinger_2ndorder_PhaseTransition(N, l0_min , l0_max, l_points, m_min, m_max, m_points)
    """
    Try to view the 1st order phase transition for varying m_lat and see how it changes from first to second order phase transition around m=0.33
    Will plot Field strength and particle number on different graphs, but many values of m_lat on the same graph
    """

    lambda = 100
    max_dim = 10
    max_sweeps = 10
    cut_off = 1e-10

    m_list = range(m_min, m_max, m_points)
    l0_list = range(l0_min, l0_max, l_points)

    p = plot(title="Particle number phase transition for varying m, \n N=$N", xlabel="Background Field Strength, θ/2π ", ylabel="Expectation of Particle Number")
    q = plot(title="Field strength phase transition for varying m, \n N=$N", xlabel="Background Field Strength, θ/2π", ylabel="Expectation of Electric Field Strength")

    x = (N/30)^2  # fix N/sqrt(x) = 30, volume of the system
    lambda = 100 

    for i in 1:m_points # vary m_lat
        particles = []
        field = []

        m_lat = m_list[i] 
    
        for j in 1:l_points # Vary l0, background field

            # Schwinger Model Variables:
            l0 = l0_list[j]

            # Create Hamiltonian MPO from one of the functions above ^^
            H_MPO, sites =  Schwinger_Wilson(N, l0, m_lat, x)

            # Call the dmrg function
            E, psi = DMRG(2*N, max_sweeps, cut_off, max_dim, H_MPO, sites)  
            println("\n")

            # Calculate expection of particle number - should be 0 for low l and then 2
            P = P_W(N, sites)
            ex_P = real(inner(psi',P,psi))
            push!(particles, ex_P)

            # Calculate expection of EFS 
            L = L_W(N, l0, sites)
            ex_L = real(inner(psi',L,psi))
            push!(field, ex_L)

        end

        # now have a list of particle number and field strength variation for a certain m
        # need to add this to the Plots
        plot!(p, l0_list, particles, label = "m=$m_lat")
        plot!(q, l0_list, field, label = "m=$m_lat")

    end

    return p, q
end
#p, q = Schwinger_2ndorder_PhaseTransition(4, 0.45, 0.6, 10, 0, 0.5, 6)
#display(p)
#display(q)
#savefig(p, "Schwinger 2nd order phase transition - Particles.pdf")
#savefig(q, "Schwinger 2nd order phase transition - Field.pdf")


println("\nFinished \n")