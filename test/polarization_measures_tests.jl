# test/polarization_measures_tests.jl
using Test
using DataFrames
using StatsBase: proportionmap
using Combinatorics: combinations

# Load from the package. If your module name differs, change "PrefPol".
import PrefPol

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"Build a ranking Dict{Symbol,Int} from an ordered vector of candidate Symbols."
ranking_dict(order::Vector{Symbol}) = Dict(c => i for (i,c) in enumerate(order))

"Repeat each ranking dict according to integer counts."
function profile_from_counts(counts::Vector{Tuple{Vector{Symbol},Int}})::Vector{Dict{Symbol,Int}}
    out = Vector{Dict{Symbol,Int}}()
    for (ord, k) in counts
        r = ranking_dict(ord)
        append!(out, fill(r, k))
    end
    out
end

"Return (paired_accum, proportion_rankings) for a profile using your pipeline."
function paired_and_props(profile)
    PrefPol.get_paired_rankings_and_proportions(profile)
end

# The canonical 3-candidate worked example (n = 29):
# p = {abc:7, acb:5, cab:1, cba:7, bca:3, bac:6}
example_counts = [
    ([:a,:b,:c], 7),
    ([:a,:c,:b], 5),
    ([:c,:a,:b], 1),
    ([:c,:b,:a], 7),
    ([:b,:c,:a], 3),
    ([:b,:a,:c], 6),
]

const EX_R = 22/29
const EX_HHI = 59/121
const EX_RHHI = sqrt(EX_R * EX_HHI)

# ---------------------------------------------------------------------------
# Tests for: find_reversal_pairs
# ---------------------------------------------------------------------------

@testset "find_reversal_pairs" begin
    unique_all = [
        ("a","b","c"), ("c","b","a"),   # pair 1
        ("a","c","b"), ("b","c","a"),   # pair 2
        ("c","a","b"), ("b","a","c")    # pair 3
    ]
    paired, unpaired = PrefPol.find_reversal_pairs(unique_all)
    @test length(paired) == 3
    # Check each tuple is (ranking, i, reversed, j) with reversal property
    @test all(reverse(p[1]) == p[3] for p in paired)
    @test isempty(unpaired)

    # One missing to force an unpaired element
    unique_missing = [
        ("a","b","c"), ("c","b","a"),   # pair ok
        ("a","c","b")                    # bca missing → unpaired
    ]
    paired2, unpaired2 = PrefPol.find_reversal_pairs(unique_missing)
    @test length(paired2) == 1
    @test length(unpaired2) == 1
    @test unpaired2[1][1] == ("a","c","b")
end

# ---------------------------------------------------------------------------
# Tests for: local_reversal_values, calc_total_reversal_component,
#            calc_reversal_HHI, fast_reversal_geometric
# ---------------------------------------------------------------------------

@testset "reversal magnitudes (R, HHI, RHHI)" begin
    profile = profile_from_counts(example_counts)
    paired, prop = paired_and_props(profile)

    # local_reversal_values should produce 14/29, 6/29, 2/29 (in some order)
    xs = collect(PrefPol.local_reversal_values(paired, prop))
    @test isapprox(sum(xs), EX_R; atol=1e-12)
    @test sort(xs) == sort([14/29, 6/29, 2/29])

    # R
    R = PrefPol.calc_total_reversal_component(paired, prop)
    @test isapprox(R, EX_R; atol=1e-12)

    # HHI = sum((x/R)^2)
    HHI = PrefPol.calc_reversal_HHI(paired, prop)
    @test isapprox(HHI, EX_HHI; atol=1e-12)

    # fast one-pass geometric should equal sqrt(sum(x^2)/sum(x)) = sqrt(R*HHI)
    rhhi_fast = PrefPol.fast_reversal_geometric(paired, prop)
    @test isapprox(rhhi_fast, EX_RHHI; atol=1e-12)
end

# ---------------------------------------------------------------------------
# Tests for: nab, dab, Ψ
# ---------------------------------------------------------------------------

@testset "nab / dab / Ψ" begin
    # Simple tiny profile (n=3)
    p1 = [
        ranking_dict([:a,:b,:c]),
        ranking_dict([:a,:c,:b]),
        ranking_dict([:b,:a,:c]),
    ]
    @test PrefPol.nab(:a, :b, p1) == 2
    @test PrefPol.nab(:b, :a, p1) == 1
    @test PrefPol.dab(:a, :b, p1) == 1

    # Unanimous profile → Ψ = 0
    unanim = fill(ranking_dict([:a,:b,:c]), 10)
    @test PrefPol.Ψ(unanim) == 0

    # Equiprobable over all 3! permutations → Ψ = 1
    allperms = [[ :a,:b,:c ], [ :a,:c,:b ], [ :b,:a,:c ],
                [ :b,:c,:a ], [ :c,:a,:b ], [ :c,:b,:a ]]
    eqprof = [ranking_dict(perm) for perm in allperms]
    @test PrefPol.Ψ(eqprof) == 1
end

# ---------------------------------------------------------------------------
# Tests for: get_paired_rankings_and_proportions & profile wrappers
#            calc_total_reversal_component(profile),
#            calc_reversal_HHI(profile),
#            fast_reversal_geometric(profile)
# ---------------------------------------------------------------------------

@testset "pairing+proportions & wrappers" begin
    profile = profile_from_counts(example_counts)
    paired, prop = paired_and_props(profile)

    # Proportions match counts / 29
    # We just check two representative rankings:
    @test isapprox(prop[("a","b","c")], 7/29; atol=1e-12)
    @test isapprox(prop[("b","a","c")], 6/29; atol=1e-12)

    # Wrappers must reproduce EX_R, EX_HHI, EX_RHHI
    @test isapprox(PrefPol.calc_total_reversal_component(profile), EX_R; atol=1e-12)
    @test isapprox(PrefPol.calc_reversal_HHI(profile), EX_HHI; atol=1e-12)
    @test isapprox(PrefPol.fast_reversal_geometric(profile), EX_RHHI; atol=1e-12)

    # Empty / no-pair case should return 0.0 (spec)
    prof_nopair = [ranking_dict([:a,:b,:c]) for _ in 1:5] # all identical → no reversals
    @test PrefPol.calc_total_reversal_component(prof_nopair) == 0.0
    @test PrefPol.calc_reversal_HHI(prof_nopair) == 0.0
    @test PrefPol.fast_reversal_geometric(prof_nopair) == 0.0  # enforces the spec/guard
end

# ---------------------------------------------------------------------------
# Tests for: consensus_for_group, kendall_tau_dict, average_normalized_distance,
#            group_avg_distance, weighted_coherence
# ---------------------------------------------------------------------------

@testset "consensus, Kendall tau, distances, coherence" begin
    # One-profile group → consensus should be that ranking
    prof = [ranking_dict([:a,:b,:c])]
    subdf = DataFrame(profile = prof)
    out = PrefPol.consensus_for_group(subdf)

    @test out == prof[1]

    # Kendall tau
    r1 = ranking_dict([:a,:b,:c])
    r2 = ranking_dict([:a,:c,:b])
    r3 = ranking_dict([:c,:b,:a])
    @test PrefPol.kendall_tau_dict(r1,r1) == 0
    @test PrefPol.kendall_tau_dict(r1,r3) == 3   # all three pairs discordant for m=3

    # average_normalized_distance
    @test PrefPol.average_normalized_distance(prof, r1) == 0.0
    @test PrefPol.average_normalized_distance(fill(r3, 5), r1) == 1.0

    # group_avg_distance → (avg_distance, group_coherence)
    gad = PrefPol.group_avg_distance(subdf)
    @test gad.avg_distance == 0.0
    @test gad.group_coherence == 1.0

    # weighted_coherence over two groups
    results_distance = DataFrame(group = ["G1","G2"], group_coherence = [1.0, 0.5])
    propmap = Dict("G1"=>0.5, "G2"=>0.5)
    @test PrefPol.weighted_coherence(results_distance, propmap, :group) == 0.75
end

# ---------------------------------------------------------------------------
# Tests for: pairwise_group_divergence, overall_divergence, overall_divergences
# ---------------------------------------------------------------------------

@testset "divergence measures" begin
    # Two groups with opposite consensus, profiles pure at their consensus
    consA = ranking_dict([:a,:b,:c])
    consB = ranking_dict([:c,:b,:a])
    profA = fill(consA, 4)
    profB = fill(consB, 6)
    m = 3

    @test PrefPol.pairwise_group_divergence(profA, consB, m) == 1.0
    @test PrefPol.pairwise_group_divergence(profB, consA, m) == 1.0

    group_profiles = Dict(:A => profA, :B => profB)
    consensus_map  = Dict(:A => consA, :B => consB)
    @test PrefPol.overall_divergence(group_profiles, consensus_map) == 1.0

    # Wrapper that uses DataFrames
    whole_df = vcat(DataFrame(group=:A, profile=profA),
                    DataFrame(group=:B, profile=profB))
    grouped_consensus = DataFrame(group=[:A,:B], x1=[consA, consB])
    @test PrefPol.overall_divergences(grouped_consensus, whole_df, :group) == 1.0
end

# ---------------------------------------------------------------------------
# Tests for: apply_measure_to_bts, apply_all_measures_to_bts
# ---------------------------------------------------------------------------

@testset "apply_measure_to_bts / apply_all_measures_to_bts" begin
    # Build two profiles with known values
    unanim = fill(ranking_dict([:a,:b,:c]), 5)
    eqprof = [ranking_dict(perm) for perm in
              ([ :a,:b,:c ], [ :a,:c,:b ], [ :b,:a,:c ],
               [ :b,:c,:a ], [ :c,:a,:b ], [ :c,:b,:a ])]

    # A "bootstrap" dictionary shaped like your pipeline expects:
    # Each variant maps to a vector of NamedTuples that have a `.profile` field.
    bts = Dict(
        :mice => [(; profile = unanim), (; profile = eqprof)],
        :rand => [(; profile = eqprof)]
    )

    # Single measure
    outΨ = PrefPol.apply_measure_to_bts(bts, PrefPol.Ψ)
    @test keys(outΨ) == keys(bts)
    @test outΨ[:mice] == [0.0, 1.0]
    @test outΨ[:rand] == [1.0]

    # All measures (subset relevant to this file)
    out_all = PrefPol.apply_all_measures_to_bts(bts;
        measures=[PrefPol.Ψ,
                  PrefPol.calc_total_reversal_component,
                  PrefPol.calc_reversal_HHI,
                  PrefPol.fast_reversal_geometric])

    @test Set(keys(out_all)) == Set(Symbol.(["Ψ",
                                     "calc_total_reversal_component",
                                     "calc_reversal_HHI",
                                     "fast_reversal_geometric"]))

    # Sanity: unanimity has no reversals ⇒ R = HHI = RHHI = 0
    @test out_all[:calc_total_reversal_component][:mice][1] == 0.0
    @test out_all[:calc_reversal_HHI][:mice][1] == 0.0
    @test out_all[:fast_reversal_geometric][:mice][1] == 0.0
end

# ---------------------------------------------------------------------------
# Tests for: compute_group_metrics, bootstrap_group_metrics
# ---------------------------------------------------------------------------

@testset "compute_group_metrics / bootstrap_group_metrics" begin
    # Two pure, coherent groups with opposite consensus
    consA = ranking_dict([:a,:b,:c])
    consB = ranking_dict([:c,:b,:a])
    df = vcat(DataFrame(group=:A, profile=fill(consA, 4)),
              DataFrame(group=:B, profile=fill(consB, 6)))

    C, D = PrefPol.compute_group_metrics(df, :group)
    @test isapprox(C, 1.0; atol=1e-12)   # within-group coherence
    @test isapprox(D, 1.0; atol=1e-12)   # strong between-group divergence

    # Bootstrap over two "replicates" per variant
    bt_profiles = Dict(
        :mice => [df, df],
        :rand => [df]
    )
    res = PrefPol.bootstrap_group_metrics(bt_profiles, :group)
    @test Set(keys(res)) == Set([:mice, :rand])
    @test res[:mice][:C] == fill(1.0, 2)
    @test res[:mice][:D] == fill(1.0, 2)
    @test res[:rand][:C] == fill(1.0, 1)
    @test res[:rand][:D] == fill(1.0, 1)
end
