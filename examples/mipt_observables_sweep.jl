import EDCircuit as ed

using LinearAlgebra
using Random
using Statistics
using Plots

#=
Sweep measurement probabilities and track MIPT observables with exact diagonalization.

NOTE: This example requires Plots.jl. If it is not installed, run:
    import Pkg; Pkg.add("Plots")
=#

Random.seed!(11)

pbc = true            # periodic boundary conditions
nsamples = 20         # number of circuit realizations per (L, p)
p_values = range(0.0, 0.3, length=6)
L_values = collect(8:2:16)
L_record = L_values[end] # system size used for measurement-record entropy vs depth
D_record = 4 * L_record  # depth for measurement-record entropy trace
p_fit = 0.15             # measurement rate for finite-size scaling of F(t)

function mutual_information(state::ed.State, sitesA::Vector{Int}, sitesB::Vector{Int})
    return ed.bip_ent_entropy(state, sitesA) + ed.bip_ent_entropy(state, sitesB) -
           ed.bip_ent_entropy(state, vcat(sitesA, sitesB))
end

function renyi2_entropy(state::ed.State, sites::Vector{Int})
    rho = ed.reduced_density_matrix(state, sites)
    purity = real(tr(rho * rho))
    return -log(purity + 1e-12)
end

function schmidt_values(state::ed.State, sites::Vector{Int}; k::Int = 4)
    # ED analogue of MPS transfer-matrix singular values:
    # Schmidt values across the chosen bipartition.
    state_tensor = reshape(state.state, state.tensordim)
    perm = ed.find_perm(sites, state)
    reshaped_state = reshape(permutedims(state_tensor, perm), (2^length(sites), 2^(state.L - length(sites))))
    values = svdvals(reshaped_state)
    return values[1:min(k, length(values))]
end

function measure_layer_with_entropy!(circuit::ed.Circuit, psi::ed.State, prob::Float64)
    rands = rand(Float64, circuit.L)
    sites = circuit.sites[findall(x -> x < prob, rands)]
    layer_entropy = 0.0
    for site in sites
        outcome_prob, _ = ed.measure!(psi, site)
        layer_entropy -= log(outcome_prob + 1e-12)
    end
    return layer_entropy
end

Sbip_avg = zeros(length(L_values), length(p_values))
S2bip_avg = zeros(length(L_values), length(p_values))
Iab_avg = zeros(length(L_values), length(p_values))
I3_avg = zeros(length(L_values), length(p_values))
mag_avg = zeros(length(L_values), length(p_values))
schmidt_gap_avg = zeros(length(L_values), length(p_values))
F_record_avg = zeros(length(p_values), D_record)
F_end_vs_L = zeros(length(L_values))

for (lidx, L) in enumerate(L_values)
    D = L
    Lbip = div(L, 2)
    Lq = div(L, 4)

    sites_bip = collect(1:Lbip)
    sitesA = collect(1:Lq)
    sitesB = collect((Lq + 1):(2 * Lq))
    sitesC = collect((2 * Lq + 1):(3 * Lq))

    circuit = ed.Circuit(pbc, L, 0)
    trotter_sites = ed.trotter_sites_vector(circuit, 2)

    for (pidx, p) in enumerate(p_values)
        Sbip_samples = Float64[]
        S2bip_samples = Float64[]
        Iab_samples = Float64[]
        I3_samples = Float64[]
        mag_samples = Float64[]
        schmidt_gap_samples = Float64[]

        for _ in 1:nsamples
            state = ed.zero_state(L)
            for _ in 1:D
                ed.apply_random_haar_gates!(state, trotter_sites)
                ed.measure_layer!(circuit, state, p)
            end

            ed.sort_sites!(state)

            push!(Sbip_samples, ed.bip_ent_entropy(state, sites_bip))
            push!(S2bip_samples, renyi2_entropy(state, sites_bip))
            push!(Iab_samples, mutual_information(state, sitesA, sitesB))
            push!(I3_samples, ed.tripartite_mutual_information(state, sitesA, sitesB, sitesC))
            push!(mag_samples, ed.magnetization(state) / L)

            schmidt_vals = schmidt_values(state, sites_bip, k=2)
            if length(schmidt_vals) >= 2
                push!(schmidt_gap_samples, schmidt_vals[1] - schmidt_vals[2])
            else
                push!(schmidt_gap_samples, schmidt_vals[1])
            end
        end

        Sbip_avg[lidx, pidx] = mean(Sbip_samples)
        S2bip_avg[lidx, pidx] = mean(S2bip_samples)
        Iab_avg[lidx, pidx] = mean(Iab_samples)
        I3_avg[lidx, pidx] = mean(I3_samples)
        mag_avg[lidx, pidx] = mean(mag_samples)
        schmidt_gap_avg[lidx, pidx] = mean(schmidt_gap_samples)
    end
end

for (pidx, p) in enumerate(p_values)
    F_samples = zeros(nsamples, D_record)
    circuit = ed.Circuit(pbc, L_record, 0)
    trotter_sites = ed.trotter_sites_vector(circuit, 2)

    for s in 1:nsamples
        state = ed.zero_state(L_record)
        cumulative_F = 0.0
        for t in 1:D_record
            ed.apply_random_haar_gates!(state, trotter_sites)
            cumulative_F += measure_layer_with_entropy!(circuit, state, p)
            F_samples[s, t] = cumulative_F
        end
    end

    F_record_avg[pidx, :] = vec(mean(F_samples, dims=1))
end

for (lidx, L) in enumerate(L_values)
    D_fit = 4 * L
    circuit = ed.Circuit(pbc, L, 0)
    trotter_sites = ed.trotter_sites_vector(circuit, 2)
    F_samples = zeros(nsamples)
    for s in 1:nsamples
        state = ed.zero_state(L)
        cumulative_F = 0.0
        for _ in 1:D_fit
            ed.apply_random_haar_gates!(state, trotter_sites)
            cumulative_F += measure_layer_with_entropy!(circuit, state, p_fit)
        end
        F_samples[s] = cumulative_F
    end
    F_end_vs_L[lidx] = mean(F_samples)
end

p1 = plot(xlabel="L", ylabel="Entropy", title="Entropies")
for (pidx, p) in enumerate(p_values)
    plot!(p1, L_values, Sbip_avg[:, pidx], label="S_A, p=$(round(p, digits=2))")
    plot!(p1, L_values, S2bip_avg[:, pidx], label="S2_A, p=$(round(p, digits=2))", ls=:dash)
end

p2 = plot(xlabel="L", ylabel="Mutual info", title="Mutual information")
for (pidx, p) in enumerate(p_values)
    plot!(p2, L_values, Iab_avg[:, pidx], label="I(A:B), p=$(round(p, digits=2))")
    plot!(p2, L_values, I3_avg[:, pidx], label="I3, p=$(round(p, digits=2))", ls=:dash)
end

p3 = plot(xlabel="L", ylabel="Magnetization / site", title="Magnetization")
for (pidx, p) in enumerate(p_values)
    plot!(p3, L_values, mag_avg[:, pidx], label="p=$(round(p, digits=2))")
end

p4 = plot(xlabel="L", ylabel="Schmidt gap", title="Schmidt gap")
for (pidx, p) in enumerate(p_values)
    plot!(p4, L_values, schmidt_gap_avg[:, pidx], label="p=$(round(p, digits=2))")
end

plt = plot(p1, p2, p3, p4, layout=(2, 2), size=(1100, 800))
savefig(plt, "mipt_observables_vs_L.png")

println("Saved plot to mipt_observables_vs_L.png")

pF = plot(xlabel="Depth", ylabel="F(t)", title="Measurement-record entropy (L=$(L_record))")
for (pidx, p) in enumerate(p_values)
    plot!(pF, 1:D_record, F_record_avg[pidx, :], label="p=$(round(p, digits=2))")
end
savefig(pF, "measurement_record_entropy_vs_t.png")
println("Saved plot to measurement_record_entropy_vs_t.png")

invL2 = 1.0 ./ (L_values .^ 2)
slope = cov(invL2, F_end_vs_L) / var(invL2)
intercept = mean(F_end_vs_L) - slope * mean(invL2)
fit_vals = slope .* invL2 .+ intercept
pFit = plot(invL2, F_end_vs_L, seriestype=:scatter, xlabel="1/L^2", ylabel="F(t_end)",
    title="F(t_end) vs 1/L^2 at p=$(p_fit)")
plot!(pFit, invL2, fit_vals, label="fit: slope=$(round(slope, digits=4)), intercept=$(round(intercept, digits=4))")
savefig(pFit, "measurement_record_entropy_fits_vs_L.png")
println("Saved plot to measurement_record_entropy_fits_vs_L.png")
println("Fit results for p=$(p_fit): slope=$(slope), intercept=$(intercept)")
