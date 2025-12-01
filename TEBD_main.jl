using ITensors, ITensorMPS
using Plots

"""
TEBD splits Hamiltonian into sets of mutually commuting gates.
Each set of gates all commute with each other so can be exponentiaited into time evolution op individually
Apply these time evolution operators seperately
Error scales with time step (squared in 2nd order case) and with value of commutator between gate sets
"""

# Ising Gates and Hamiltonian
function get_exp_Hzz_gate_list(sites, a, J, hz)

    gates = ITensor[] # important to specify the type of the elements of this list

    # Write a for loop to create the gates list for ZZ and Z terms
    # Useful things here are the exp function, the op function e.g. op("Z", sites[n])
    # The input a is -1im*tau for the standard time step but sometimes we might want to 
    # use -1im*tau/2 for higher order schemes - will do 2nd order here
    N = length(sites)

    for i in 1:N
        # we can do this more efficiently in a list of length N rather than 2N

        # ZZ terms
        if i != N  # Dont do it for the last term or will be out of bounds
            gate1 = op( "Z", sites[i]) 
            gate2 = op("Z", sites[i+1]) 
            ZZ = J * gate1 * gate2

            expZZ = exp(a * ZZ /2)
            push!(gates, expZZ)
        end

        # Z terms
        Z = op( "Z", sites[i])  
        Z = hz * Z

        expZ = exp(a* Z /2)
        push!(gates, expZ)

    end

    return gates
end

function get_exp_Hx_gate_list(sites, a, hx)

    gates = ITensor[]

    # Similar to the function above for the X terms
    N = length(sites)

    for i in 1:N
        X = op("X", sites[i])  # X gate as site i
        X = hx * X

        expX = exp(a*X)
        push!(gates, expX)
    end

    return gates
end

function get_H(sites, J, hz, hx)

    # get the Hamiltonian MPO so that we can compute the energy observable
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

    # Define the Hamiltonian MPO with OpSum and MPO constructor:
    os = OpSum()
    # Transverse Field Ising Model Hamiltonian: H = -J sum(Sz_j Sz_j+1) - gx sum(Sx_j) - gz sum(Sz_j)

    # Nearest-neighbor ZZ term
    for j in 1:(N-1)
        add!(os, J, "Z", j, "Z", j+1)
    end
    # Transverse field hx * X_i
    for j in 1:N
        add!(os, hx, "X", j)  
    end
    # Longitudinal field hz * Z_i
    for j in 1:N
        add!(os, hz, "Z", j)
    end

    H_MPO = MPO(os, sites)
    return H_MPO
end

# Schwinger Gates and Hamiltonian - Schwinger working correctly in Schwinger_TEBD file
function get_Schwinger_odd(sites, N, l0, m_lat, x, a)

    r = 1

    gates = ITensor[]

    for i in 0:(N-1)
        if iseven(i)  # i 0 indexed
            if i != (N-1)

                X1 = op("X", sites[2*i+2])  # X gate as site 2n+1  - not working, accesses out of bounds
                X2 = op("X", sites[2*i+3])

                Y1 = op("Y", sites[2*i+2])
                Y2 = op("Y", sites[2*i+3])
            end

            X3 = op("X", sites[2*i+1])
            X4 = op("X", sites[2*i+2])

            Y3 = op("Y", sites[2*i+1])
            Y4 = op("Y", sites[2*i+2])

            if i == (N-1)
                gate = (m_lat*sqrt(x) + x*r)*(X3*X4 + Y3*Y4)  # no X1 etc
            else
                gate1 = (x*(r+1)/2)*(X1*X2 + Y1*Y2)
                gate2 = (m_lat*sqrt(x) + x*r)*(X3*X4 + Y3*Y4)
                gate = apply(gate1, gate2)
            end

            expX = exp(a*gate/2)  # exponentiate a/2

            push!(gates, expX)
        end
    end

    for n in 0:(2N-3)
        if iseven(n)
            Z = op("Z", sites[n+1])  # Z terms
            Z = l0*(N-ceil((n+1)/2))*Z
            expZ = exp(a*Z/2)
            push!(gates, expZ)
        end
    end

    return gates
end

function get_Schwinger_even(sites, N, l0, m_lat, x, a)

    r = 1
    lambda = 0

    gates = ITensor[]

    for i in 0:N-1
        if isodd(i)  # i 0 indexed
            if i != N-1
                X1 = op("X", sites[2*i+2])  # X gate as site 2n+1
                X2 = op("X", sites[2*i+3])

                Y1 = op("Y", sites[2*i+2])
                Y2 = op("Y", sites[2*i+3])
            end

            X3 = op("X", sites[2*i+1])
            X4 = op("X", sites[2*i+2])

            Y3 = op("Y", sites[2*i+1])
            Y4 = op("Y", sites[2*i+2])

            if i == N-1
                gate = (m_lat*sqrt(x) + x*r)*(X3*X4 + Y3*Y4)  # no X1 etc
            else
                gate1 = (x*(r+1)/2)*(X1*X2 + Y1*Y2)
                gate2 = (m_lat*sqrt(x) + x*r)*(X3*X4 + Y3*Y4)
                gate = apply(gate1, gate2)
            end

            if i == 1
                # Constant Energy Shift
                E0 = (l0^2)*(N-1)+ 0.25*N*(N-1) + lambda*N/2
                Id = op("Id", sites[2*i+1])
                Id =  E0*Id
                gate = apply(gate, Id)
            end

            expX = exp(a*gate)  # exponentiate
            
            push!(gates, expX)
        end
    end

    for n in 0:(2N-3)
        if isodd(n)
            Z = op("Z", sites[n+1])  # Z terms
            Z = l0*(N-ceil((n+1)/2))*Z
            expZ = exp(a*Z)
            push!(gates, expZ)
        end
    end

    return gates
end

function get_Schwinger_zz(sites, a, N, l0) 
    # a = - tau
    # e^(aH_zz/2)  = 1 + aH_ZZ/2 + 1/2(aH_zz/2)^2
    lambda = 0

    op = OpSum()
    op += ("Id", 1)
    MPO1 = MPO(op, sites)  # identity

    os = OpSum()

    for n in 0:(2N-1)
        for k in (n+1):(2N-1)
            add!(os, 0.5*(N-ceil((k+1)/2)+lambda) , "Z", n+1, "Z", k+1)
        end
    end

    MPO2 = MPO(os, sites)  # H_zz
    MPO3 = apply(MPO2, MPO2) # H_zz^2

    MPO4 = MPO1 + (a/2)*MPO2  + 0.5*((a/2)^2)*MPO3  # I + a/2 H_zz + a^2/8 H_zz^2
    return MPO4
end

function Schwinger_Wilson(sites, N, l0, m_lat, x)
    """
    Creates Wilson Hamiltonian MPO for the Schwinger model as found in eq4.2 Takis Thesis
    Inputs
        N - Number of sites
        theta - Topological term between 0-2pi
        g - Coupling constant
        m_lat - lattice mass
        
    """
    # Define sites object using siteinds
    #sites = ITensors.siteinds("S=1/2", 2*N)  # need 2N sites for N physical

    # Define the Hamiltonian MPO with OpSum and MPO constructor:
    os = OpSum()

    # Variables
    r = 1  # Wilson Parameter, usually 1 
    g = 1
    # a =  1 # Lattice Spacing 
    # x = 1/(a*g)^2
    # l0 = theta/(2*pi)
    lambda = 0 # Lagrange multiplier that must be 'sufficiently large' - ensures no ground state degeneracy
    
    
    # eq 1.27 
    for n in 0:(N-2)
        # add!(os, x*(r-1) , "S+", 2n+1, "Z", 2n+2, "Z", 2n+3, "S-", 2n+4)  
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
    return H_MPO
end

# TEBDs
function LTFIM_TEBD_2nd_order(steps, mps, tau, J, hx, hz, cutoff, maxdim, H, sites)
    # TEBD for Ising, applies Zs and Xs seperately

    N = length(mps)

    # Get the gate lists using the functions above
    U1_list = get_exp_Hzz_gate_list(sites, tau, J, hz)
    U2_list = get_exp_Hx_gate_list(sites, tau, hx)

    # Create Observables to be tracked 
    Energy_Oberservable = get_H(sites, J, hz, hx)  # The Hamiltonian

    # Initialize the lists of the observables you want to track
    Magnetisation_list = []
    Energy_list = []
    Entropy_list =[]
    Correlations_list = []

    # Write a for loop over the time steps for the time evolution
    # At each step we need to apply the lists to the mps using e.g. mps = apply(He, mps; cutoff = cutoff, maxdim = maxdim)
    # At each step you can compute the observables you want to track
    for i in 1:steps

        # Time Evolve with operators
        mps = apply(U1_list, mps; cutoff = cutoff, maxdim = maxdim) # First U1
        mps = apply(U2_list, mps; cutoff = cutoff, maxdim = maxdim)
        mps = apply(U1_list, mps; cutoff = cutoff, maxdim = maxdim) # Second U1

        mps = normalize(mps)  # Especially necessary for imaginary time evolution
        
        # Observables
        Magnetisation = expect(mps, "Sz")
        push!(Magnetisation_list, Magnetisation)

        Energy = inner(mps', Energy_Oberservable, mps)
        push!(Energy_list, Energy)

        Entanglement_Entropy_Observable = Entanglement_Entropy_spectrum(mps, N)
        push!(Entropy_list, Entanglement_Entropy_Observable)

        Correlations = Correlation(mps, N)
        push!(Correlations_list, Correlations)

    end

    # Return MPS and lists of observables you kept track of
    return mps, Magnetisation_list, Energy_list, Entropy_list, Correlations_list
end

function TEBD_Schwinger(steps, mps, sites, N, l0, m_lat, x, a, maxdim, cutoff)
    # TEBD for Schwinger

    # Get the gate lists using the functions above
    U1_list = get_Schwinger_odd(sites, N, l0, m_lat, x, a)
    U2_MPO = get_Schwinger_zz(sites, a, N, l0)
    U3_list = get_Schwinger_even(sites, N, l0, m_lat, x, a)

    # Create Observables to be tracked 
    Energy_Oberservable = Schwinger_Wilson(sites, N, l0, m_lat, x)  # The Hamiltonian

    # Initialize the lists of the observables you want to track
    Energy_list = []


    # Write a for loop over the time steps for the time evolution
    # At each step we need to apply the lists to the mps using e.g. mps = apply(He, mps; cutoff = cutoff, maxdim = maxdim)
    # At each step you can compute the observables you want to track
    for i in 1:steps

        # Time Evolve with operators
        #mps = apply(U1_list, mps; cutoff = cutoff, maxdim = maxdim) # First U1
        mps = apply(U2_MPO, mps; cutoff = cutoff, maxdim = maxdim)  # ZZ MPO
        #mps = apply(U3_list, mps; cutoff = cutoff, maxdim = maxdim)
        mps = apply(U2_MPO, mps; cutoff = cutoff, maxdim = maxdim) # ZZ MPO
        #mps = apply(U1_list, mps; cutoff = cutoff, maxdim = maxdim) # Second U1

        mps = normalize(mps)  # Especially necessary for imaginary time evolution
        
        # Observables
        Energy = inner(mps', Energy_Oberservable, mps)
        push!(Energy_list, Energy)

    end

    # Return MPS and lists of observables you kept track of
    return mps, Energy_list
end

# Analysis:
function Entanglement_Entropy_spectrum(psi, N)
    """
    Finds the entanglement entropy of an mps
    Evaluates it at each site and adds to a list
    """
    Entanglement_spectrum = []
    for b in 1:N
        psi_N = orthogonalize!(psi, b)
        U,S,V = svd(psi_N[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
        SvN = 0.0
        for n=1:dim(S, 1)
            p = S[n,n]^2
            SvN -= p * log(p)
        end
        push!(Entanglement_spectrum, SvN)
    end
    return Entanglement_spectrum
end

function Correlation(psi, N)
    """
    Calculates the correlation function Ci,j​(t)=⟨σiz​σjz​⟩−⟨σiz​⟩⟨σjz​⟩ from the center of the MPS to each site and returns a list
    Can just use the correlation_matrix() function from ITensors
    """

    C = correlation_matrix(psi, "Z", "Z")
    #println(C)
    Correlations = C[Int(round(N/2)), :]  # just take the correlations with the center
    return Correlations
end

# Applications:
function Ising_Quenches(N, J_init, J_final, hx_init, hx_final, hz, tau)

    """
    Creates the Ising model Hamiltonian for the initial parameters and finds ground state
    Then creates quenched hamiltonian and time evolves the state 
    Tracks Observables and returns plots
    """

    # Parameters for time evolution
    a = -im*tau 
    cutoff = 1e-10
    steps = 100
    sweeps = 10
    maxdim = 20

    # Finds time evolution of Ising model when quenched (Sudden limit), also do for adiabatic limit?
    sites = siteinds("S=1/2", N)
    H_initial = get_H(sites, J_init, hz, hx_init)  # Initial Hamiltonian
    H = get_H(sites, J_final, hz, hx_final)  # Final (Quenched) Hamiltonian

    mps = randomMPS(sites)
    # mps = MPS(sites, n -> isodd(n) ? "Up" : "Dn")
    # find ground state of intitial Hamiltonian
    energy, mps = dmrg(H_initial, mps, nsweeps = sweeps, cutoff = cutoff, maxdim = maxdim, outputlevel = 0)

    # Then turn on the external fields - Quench it - and time evolve
    psi, Magnetisation_list, Energy_list, Entropy_list = LTFIM_TEBD_2nd_order(steps, mps, a, J, hx, hz, cutoff, maxdim, H, sites)

    # Plots:
    magnetisation_array = reduce(hcat, Magnetisation_list)  # turn into a matrix for plotting
    M = heatmap(magnetisation_array, xlabel="Time Step", ylabel= "Site Index", colourbar_title = "Magnetisation")

    entropy_array = reduce(hcat, Entropy_list)  # turn into a matrix for plotting
    EE = heatmap(entropy_array, xlabel="Time Step", ylabel= "Site Index", colourbar_title = "Entanglement Entropy")

    E0 = plot(real.(Energy_list))

    # Return the final state and plots
    return mps, M, EE, E0
end

function Spinons(N, J, hx, hz, tau)

    """
    Creates the Ising model Hamiltonian for the initial parameters and finds ground state
    Then flips one spin to see two spinons propagate
    """

    # Parameters for time evolution
    a = -im*tau 
    cutoff = 1e-10
    steps = 500
    sweeps = 10
    maxdim = 20


    sites = siteinds("S=1/2", N)
    H = get_H(sites, J, hz, hx)  # Hamiltonian

    mps = randomMPS(sites)
    # mps = MPS(sites, n -> isodd(n) ? "Up" : "Dn")
    # find ground state of intitial Hamiltonian
    energy, mps = dmrg(H, mps, nsweeps = sweeps, cutoff = cutoff, maxdim = maxdim, outputlevel = 0)

    # Now apply X gate to one site in the middle
    X = op("X", sites[Int(round(N/2))])
    psi = apply(X, mps)

    # time evolve with the excitation
    psi, Magnetisation_list, Energy_list, Entropy_list, correlations_list = LTFIM_TEBD_2nd_order(steps, psi, a, J, hx, hz, cutoff, maxdim, H, sites)

    # Plots:
    magnetisation_array = reduce(hcat, Magnetisation_list)  # turn into a matrix for plotting
    M = heatmap(magnetisation_array, xlabel="Time Step", ylabel= "Site Index", colourbar_title = "Magnetisation")

    correlations_array = real.(reduce(hcat, correlations_list))  # turn into a matrix for plotting
    C = heatmap(correlations_array, xlabel="Time Step", ylabel= "Site Index", colourbar_title = "Correlation coefficient to center")

    # Return the final state and plots
    return psi, M, C
end

function Imaginary_Time(N, J, hx, hz, tau; Schwinger = false)
    # Test TEBD using imaginary time evolution - compare to DMRG energy
    a = -tau

    # Parameters for time evolution
    cutoff = 1e-10
    steps = 1000
    sweeps = 10
    maxdim = 20

    # Finds time evolution of Ising model when quenched (Sudden limit), also do for adiabatic limit?
    
    if Schwinger
        sites = siteinds("S=1/2", 2*N)
        l0 = 0
        m_lat = 0
        x = (N/40)^2

        H = Schwinger_Wilson(sites, N, l0, m_lat, x)  # get schwinger hamiltonian
    
        mps = randomMPS(sites) # Random MPS for intial guess

        # find ground state of Hamiltonian with dmrg
        energy_dmrg, psi = dmrg(H, mps, nsweeps = sweeps, cutoff = cutoff, maxdim = maxdim, outputlevel = 0)

        # find ground state with imaginary time evolution TEBD
        mps, Energy_list = TEBD_Schwinger(steps, mps, sites, N, l0, m_lat, x, a, maxdim, cutoff)
        energy_tebd = last(Energy_list)
    else

        sites = siteinds("S=1/2", N)
        H = get_H(sites, J, hz, hx)  # Ising Hamiltonian

        mps = randomMPS(sites) # Random MPS for intial guess
        # find ground state of Hamiltonian with dmrg
        energy_dmrg, psi = dmrg(H, mps, nsweeps = sweeps, cutoff = cutoff, maxdim = maxdim, outputlevel = 0)

        mps, Magnetisation_list, Energy_list, Entropy_list = LTFIM_TEBD_2nd_order(steps, mps, a, J, hx, hz, cutoff, maxdim, H, sites)
        energy_tebd = last(Energy_list)
    end

    # find the difference between both final energies
    difference = energy_dmrg - energy_tebd

    # plot the difference at each step
    difference_list = (real.(Energy_list .- energy_dmrg))
    # difference_list = difference_list./real.(energy_dmrg)
    p = plot(difference_list, xlabel="Time Step", ylabel="Energy difference", title="Log10 difference between DMRG energy\n and imaginary time evolution TEBD Energy\n N=$N, tau=$tau", legend=false)
    
    return difference, p
end

# TEBD variables
N = 4
J = 1.0
hx = 1.7
hz = 0.01
tau = 0.005

J_init = J
J_final = J
hx_init = 0.0

hx_final = hx


#mps, M, EE, E0 = Ising_Quenches(N, J_init, J_final, hx_init, hx_final, hz, tau)
#psi, M, C = Spinons(N, J, hx, hz, tau)
difference, p = Imaginary_Time(N, J, hx, hz, tau; Schwinger=true)

display(p)
#savefig(M, "Magentisation of spinons in Ising, hx =0.7.pdf")
#savefig(C, "Correlation of spinons in Ising hx =0.7.pdf")

println("\nFinished.")