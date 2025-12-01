using ITensors, ITensorMPS
using Plots

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
    # just looking at odd
    for n in 0:(N-2)  # works
        add!(os, 0.5*x*(r+1) , "X", 2n+2, "X", 2n+3)
        add!(os, 0.5*x*(r+1) , "Y", 2n+2, "Y", 2n+3)
    end
    
    coeff = (m_lat/g)*sqrt(x) + x*r  # works
    for n in 0:N-1
        add!(os, coeff , "X", 2n+1, "X", 2n+2)
        add!(os, coeff , "Y", 2n+1, "Y", 2n+2)
    end

    # ZZ - works
    for n in 0:(2N-1)
        for k in (n+1):(2N-1)
            add!(os, 0.5*(N-ceil((k+1)/2)+lambda) , "Z", n+1, "Z", k+1)
        end
    end
    
    # Z terms - works
    for n in 0:(2N-3)
        add!(os, l0*(N-ceil((n+1)/2)) , "Z", n+1)
    end
    
    # Constant Energy Shift - works
    E0 = (l0^2)*(N-1)+ 0.25*N*(N-1) + lambda*N/2
    #add!(os, E0, "Id", 1)
    
    
    H_MPO = MPO(os, sites)
    return H_MPO
end

function get_Schwinger_odd(sites, N, l0, m_lat, x, a)

    r = 1

    gates = ITensor[]

    for i in 1:N-1  # Just the second sum

        X3 = op("X", sites[2*i+1])  
        X4 = op("X", sites[2*i+2])

        Y3 = op("Y", sites[2*i+1])
        Y4 = op("Y", sites[2*i+2])

        coeff = (m_lat)*sqrt(x) + x*r

        gate2 = (coeff)*(X3*X4 + Y3*Y4)  

        expX = exp(a*gate2 /2)  # exponentiate
        
        push!(gates, expX)
    end
    

    for n in 0:(2N-3)
        if iseven(n)
            Z = op("Z", sites[n+1])  # Z terms - works
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

    for i in 0:(N-2)    # just the first sum

        #if iseven(i)  # i 0 indexed
        #if i != (N-1)
            # works
        X1 = op("X", sites[2*i+2])  
        X2 = op("X", sites[2*i+3])

        Y1 = op("Y", sites[2*i+2])
        Y2 = op("Y", sites[2*i+3])

        gate1 = (x*(r+1)/2)*(X1*X2 + Y1*Y2)
     
        expX = exp(a*gate1)  # exponentiate
        
        push!(gates, expX)
    
        
    end

    # Constant Energy Shift - works
    E0 = (l0^2)*(N-1)+ 0.25*N*(N-1) + lambda*N/2
    Id = op("Id", sites[2])
    Id =  E0*Id
    push!(gates , Id)

    # Z terms - works
    for n in 0:(2N-3)
        if isodd(n)
            Z = op("Z", sites[n+1])  
            Z = l0*(N-ceil((n+1)/2))*Z
            expZ = exp(a*Z)
            push!(gates, expZ)
        end
    end

    return gates
end

function get_Schwinger_zz(sites, a, N, l0) # works
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

function TEBD_Schwinger(steps, mps, sites, N, l0, m_lat, x, a, maxdim, cutoff)


    # Get the gate lists using the functions above
    U1_list = get_Schwinger_odd(sites, N, l0, m_lat, x, a)
    U2_MPO = get_Schwinger_zz(sites, a, N, l0)
    U3_list = get_Schwinger_even(sites, N, l0, m_lat, x, a)

    # Create Observables to be tracked 
    Energy_Oberservable = Schwinger_Wilson(sites, N, l0, m_lat, x)  # The Hamiltonian
    Particle_Observable = P_W(N, sites)
    Field_Observable = L_W(N, l0, sites)


    # Initialize the lists of the observables you want to track
    Energy_list = []
    Particle_list = []
    Field_list = []


    # Write a for loop over the time steps for the time evolution
    # At each step we need to apply the lists to the mps using e.g. mps = apply(He, mps; cutoff = cutoff, maxdim = maxdim)
    # At each step you can compute the observables you want to track
    for i in 1:steps

        # Time Evolve with operators
        mps = apply(U1_list, mps; cutoff = cutoff, maxdim = maxdim) # Odd
        mps = apply(U2_MPO, mps; cutoff = cutoff, maxdim = maxdim)  # ZZ MPO
        mps = apply(U3_list, mps; cutoff = cutoff, maxdim = maxdim)  # Even 
        mps = apply(U2_MPO, mps; cutoff = cutoff, maxdim = maxdim) # ZZ MPO
        mps = apply(U1_list, mps; cutoff = cutoff, maxdim = maxdim) # Odd

        mps = normalize(mps)  # Especially necessary for imaginary time evolution
        
        # Observables
        Energy = inner(mps', Energy_Oberservable, mps)
        push!(Energy_list, Energy)

        Particles = inner(mps', Particle_Observable, mps)
        push!(Particle_list, Particles)

        Field = inner(mps', Field_Observable, mps)
        push!(Field_list, Field)

    end

    # Return MPS and lists of observables you kept track of
    return mps, Energy_list, Particle_list, Field_list
end

# Test TEBD using imaginary time evolution - compare to DMRG energy
N = 4
tau = 0.05
#a = -tau
a = im*tau

# Parameters for time evolution
cutoff = 1e-14
steps = 250
sweeps = 10
maxdim = 20

sites = siteinds("S=1/2", 2*N)

# Parameters for Schwinger
l0 = 1.0
m_lat = 5  # -> infinity
m_lat_final = 0.33  # want to change
x = (N/40)^2

"""
H = Schwinger_Wilson(sites, N, l0, m_lat, x)  # get schwinger hamiltonian

mps = randomMPS(sites) # Random MPS for intial guess

# find ground state of Hamiltonian with dmrg
energy_dmrg, psi = dmrg(H, mps, nsweeps = sweeps, cutoff = cutoff, maxdim = maxdim, outputlevel = 0)

# find ground state with imaginary time evolution TEBD
mps, Energy_list = TEBD_Schwinger(steps, mps, sites, N, l0, m_lat, x, a, maxdim, cutoff)
energy_tebd = last(Energy_list)

# find the difference between both final energies
difference = energy_dmrg - energy_tebd

# plot the difference at each step
difference_list = log10.(real.(Energy_list .- energy_dmrg))
# difference_list = difference_list./real.(energy_dmrg)
p = plot(difference_list, xlabel="Time Step", ylabel="Energy difference", title="Log10 difference between DMRG energy\n and imaginary time evolution TEBD Energy\n N=$N, tau=$tau", legend=false)
display(p)

println(energy_dmrg)
println(energy_tebd)
"""
l0_list = [0.4,0.6,0.8,1.0]
l0_list = collect(0:0.2:3)
max_vals = []
#p = plot(xlabel = "External Field", ylabel="Particle number")

for i in 1:length(l0_list)
    #m_lat = m_lat_list[i]
    l0 = l0_list[i]
    # now do quenching from l0=0 -> l0 large
    H_init = Schwinger_Wilson(sites, N, 0, m_lat, x)
    # find ground state of Hamiltonian with dmrg:
    mps = randomMPS(sites) # Random MPS for intial guess
    energy_dmrg, psi = dmrg(H_init, mps, nsweeps = sweeps, cutoff = cutoff, maxdim = maxdim, outputlevel = 0)

    # Time evolve under new Hamiltonian with field:
    mps, Energy_list, Particle_list, Field_list = TEBD_Schwinger(steps, psi, sites, N, l0, m_lat, x, a, maxdim, cutoff)
    particle = maximum(real.(Particle_list))
    push!(max_vals, particle)
    #plot!(p, real.(Particle_list), label="Field = $l0")
end
#savefig(p, "Schwinger pair production, small mass.pdf")
#display(p)
p = plot(l0_list, max_vals, xlabel = "External Field", ylabel="Maximum Particle number expectation", legend=false)
display(p)
savefig(p, "Schwinger pair production, function of field, m=5 2.pdf")
println("\nFinished")