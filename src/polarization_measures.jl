
"""
    find_reversal_pairs(unique_rankings)

Given a collection of unique rankings, identify those that are reversals of one
another.

# Arguments
- `unique_rankings::Vector{<:NTuple{N,String}}`: list of unique rankings where
  each ranking is an N-tuple of candidate identifiers.

# Returns
`paired_accum, unpaired_accum`
- `paired_accum`: vector of tuples `(ranking, idx, reversed, reversed_idx)` for
  each reversal pair.
- `unpaired_accum`: vector of `(ranking, idx)` for rankings without a reversal
  in the collection.
"""
function find_reversal_pairs(unique_rankings::Vector{<:NTuple{N, String}}) where {N}
    paired_accum = Vector{Tuple{NTuple{N,String}, Int, NTuple{N,String}, Int}}()
    unpaired_accum = Vector{Tuple{NTuple{N,String}, Int}}()
    paired_indices = Set{Int}()

    for (i, ranking) in enumerate(unique_rankings)
        # Skip if already paired
        if i in paired_indices
            continue
        end

        # Reverse the ranking
        rev_ranking = reverse(ranking)
        found_index = nothing

        # Search for the reversed ranking in the remaining items
        for j in (i+1):length(unique_rankings)
            if j in paired_indices
                continue
            end
            if unique_rankings[j] == rev_ranking
                found_index = j
                break
            end
        end

        if isnothing(found_index)
            # No reversal found – record as unpaired
            push!(unpaired_accum, (ranking, i))
        else
            # Found a reversal – record the pair and mark both indices as paired
            push!(paired_accum, (ranking, i, rev_ranking, found_index))
            push!(paired_indices, i)
            push!(paired_indices, found_index)
        end
    end

    return paired_accum, unpaired_accum
end

# paired, unpaired = find_reversal_pairs(unique_top_rankings)


"""Return an iterator of local reversal values (2 * min(prop_i, prop_j)) 
   for each reversal pair in `paired_accum`."""
function local_reversal_values(
    paired_accum::Vector{<:Tuple{NTuple{N,String},Int,NTuple{N,String},Int}},
    proportion_rankings::Dict{NTuple{N,String},Float64}
) where N
    return (
        2 * min(proportion_rankings[p[1]], proportion_rankings[p[3]])
        for p in paired_accum
    )
end

"""Sum of local reversal components: Σ (2 * min(prop_i, prop_j))"""
function calc_total_reversal_component(paired_accum, proportion_rankings::Dict)
    sum(local_reversal_values(paired_accum, proportion_rankings))
end



"""Sum of squares of local reversal components: Σ (2 * min(prop_i, prop_j))^2"""
function calc_reversal_HHI(paired_accum, proportion_rankings::Dict)
    loc_revs = local_reversal_values(paired_accum, proportion_rankings)
    total_R = sum(loc_revs)
    sum(x^2 for x in (loc_revs ./ total_R))
end



"""Faster one-pass geometric measure: 
   accumulate sum and sum of squares in one loop, then sqrt(...)"""
function fast_reversal_geometric(paired_accum, proportion_rankings::Dict)
    total = 0.0; sumsq = 0.0
    for x in PrefPol.local_reversal_values(paired_accum, proportion_rankings)
        total += x; sumsq += x*x
    end
    return total == 0 ? 0.0 : sqrt(sumsq/total)
end


"""
    nab(candidate1, candidate2, profile)

Count the number of rankings in `profile` where `candidate1` is ranked above
`candidate2`.
"""
function nab(candidate1, candidate2, profile::Vector{<:Dict})
    return count(ranking -> ranking[candidate1] < ranking[candidate2], profile)
end

"""
    dab(candidate1, candidate2, profile)

Return the absolute difference in counts of rankings that prefer `candidate1`
over `candidate2` and vice versa. A higher value indicates greater disagreement
between the two candidates across the profile.
"""
function dab(candidate1, candidate2, profile::Vector{<:Dict})
    return abs(nab(candidate1, candidate2, profile) - nab(candidate2, candidate1, profile))
end



"""
    Ψ(p)

Compute the Can polarization measure Ψ for the profile `p`.

`p` is a vector of ranking dictionaries. The measure averages, over all pairs
of candidates, how often the rankings disagree on their relative ordering. The
result lies in `[0,1]` with larger values indicating greater polarization.
"""
function Ψ(p)
    cs = collect(keys(p[1]))
    candidate_pairs = collect(combinations(cs,2))

    m_choose_2 = length(candidate_pairs)
    n = length(p)

    ∑ = sum 

    can_polarization = ∑([(n-dab(c1,c2,p)) for (c1,c2) in candidate_pairs])/(n*m_choose_2)
    return(can_polarization)

end # TODO: note this function can be applied even to incomplete preferences, since only operates on pairs



"""
    get_paired_rankings_and_proportions(profile)

Convert a profile of rankings into reversal pairs and associated proportions.

Returns
```
paired, proportion_rankings
```
where `paired` is the output of [`find_reversal_pairs`] and
`proportion_rankings` maps each ranking tuple to its proportion in the
profile.
"""
function get_paired_rankings_and_proportions(profile)
    
    tupled = map(x->collect(x) .|> Tuple , profile)
    sorted_tupled = map(x->sort(x; by = x -> x[2]), tupled)
    rankings = [Tuple([string(x[1]) for x in ranking_origin]) for ranking_origin in sorted_tupled]
    
    unique_rankings = unique(rankings)
    
    test_paired, _ = find_reversal_pairs(unique_rankings)
    proportion_test_rankings = proportionmap(rankings)

    return(test_paired, proportion_test_rankings)
end


    
"""
    calc_total_reversal_component(profile)

Convenience wrapper that computes the total reversal component directly from a
profile of rankings.

# Arguments
- `profile`: collection of ranking dictionaries.

# Returns
Total reversal component as a `Float64`.
"""
function calc_total_reversal_component(profile)
         paired, proportion_rankings = get_paired_rankings_and_proportions(profile)
          return isempty(paired) ? 0.0 : calc_total_reversal_component(paired, proportion_rankings)
end

"""
    calc_reversal_HHI(profile)

Compute the reversal Herfindahl–Hirschman Index for a profile of rankings. This
calls [`calc_reversal_HHI(paired_accum, proportion_rankings)`] after forming the
reversal pairs and proportions.
"""
function calc_reversal_HHI(profile)
    paired, proportion_rankings =  get_paired_rankings_and_proportions(profile)
    return isempty(paired) ? 0.0 : calc_reversal_HHI(paired, proportion_rankings)
end

"""
    fast_reversal_geometric(profile)

Compute the geometric reversal measure for a profile using the fast one-pass
algorithm. Returns zero when the profile has no reversal pairs.
"""
function  fast_reversal_geometric(profile)
    paired, proportion_rankings =  get_paired_rankings_and_proportions(profile)
    geometric_reversal = fast_reversal_geometric(paired, proportion_rankings)
    return(geometric_reversal)
end

"""
    consensus_for_group(subdf)

Extract the consensus ranking for a demographic subgroup represented by
`subdf`.

`subdf` is a subset of a `DataFrame` containing a `profile` column of ranking
dictionaries. The function returns the consensus_rankings dict.
"""
function consensus_for_group(subdf)
    # subdf is a subset of df for a particular (religion, race) group
    # Extract the profiles from this sub-dataframe
    group_profiles = collect(subdf.profile)  # Vector{Dict{Symbol, Int}}

    # Call your existing function
    consensus_back_from_permallows, consensus_dict = get_consensus_ranking(group_profiles)
    #println(consensus_back_from_permallows)
    # Return as a NamedTuple so DataFrames can handle it easily
    return (consensus_ranking = consensus_dict)
end



"""
    kendall_tau_dict(r1::Dict{T,Int}, r2::Dict{T,Int}) where T

Given two ranking dictionaries (mapping candidate → rank, with lower numbers = better),
returns the Kendall tau distance (number of discordant candidate pairs).

Two candidates (a, b) are discordant if
    (r1[a] < r1[b]) != (r2[a] < r2[b]).
"""
function kendall_tau_dict(r1::Dict{T,Int}, r2::Dict{T,Int}) where T
    keys_ = collect(keys(r1))
    n = length(keys_)
    d = 0
    for i in 1:(n-1)
        for j in i+1:n
            a = keys_[i]
            b = keys_[j]
            if ( (r1[a] < r1[b]) != (r2[a] < r2[b]) )
                d += 1
            end
        end
    end
    return d
end

"""
    average_normalized_distance(profile, consensus)

Given a profile (a Vector of Dict{Symbol,Int}) and a consensus ranking `consensus` (also a Dict),
computes the average normalized Kendall tau distance between each ranking in the profile and `consensus`.

The normalization factor is binomial(m,2) where m is the number of alternatives.
"""
function average_normalized_distance(profile, consensus)
    n = length(profile)
    m = length(consensus)
    norm_factor = binomial(m, 2)
    
    dtau = kendall_tau_dict

    total_distance = sum(dtau(ranking, consensus) for ranking in profile)
    return total_distance / (n * norm_factor)
end

"""
    group_avg_distance(subdf)

Compute the average normalized distance and coherence for a demographic
subgroup.

`subdf` is a subset of a `DataFrame` for a particular group and must contain a
`profile` column of ranking dictionaries.

Returns a named tuple with fields `avg_distance` and `group_coherence`.
"""
function group_avg_distance(subdf)
    # Convert subdf.profile (a SubArray) to a plain Vector of dictionaries.
    group_profiles = collect(subdf.profile)
    # Compute the consensus ranking for this group.
    _, consensus_dict = get_consensus_ranking(group_profiles)
    # Compute the average normalized distance.
    avg_dist = average_normalized_distance(group_profiles, consensus_dict)
    # Compute group coherence.
    group_coherence = 1.0 - avg_dist
    return (avg_distance = avg_dist, group_coherence = group_coherence)
end


"""
    weighted_coherence(results_distance, proportion_map, key)

Compute weighted coherence across demographic groups.

# Arguments
- `results_distance::DataFrame`: contains `group_coherence` column produced by
  [`group_avg_distance`].
- `proportion_map::Dict`: mapping from group identifier to its proportion in
  the population.
- `key`: column symbol identifying the group in `results_distance`.
"""
function weighted_coherence(results_distance::DataFrame, proportion_map::Dict, key)
    total = sum(row.group_coherence * proportion_map[row[key]] for row in eachrow(results_distance))

    # Cstar = (2*total - 1)
    # 3) floor at zero (any tiny numerical dips below 0 become exactly zero)
    #return max(Cstar, 0.0)
    return total
end


"""
    pairwise_group_divergence(profile_i, consensus_j, m)

Average normalized Kendall tau distance between every ranking in `profile_i`
and consensus ranking `consensus_j`.

# Arguments
- `profile_i`: vector of ranking dictionaries for group `i`.
- `consensus_j`: consensus ranking dictionary for group `j`.
- `m::Int`: number of alternatives.
"""
function pairwise_group_divergence(profile_i, consensus_j, m::Int)
    n_i = length(profile_i)
    norm_factor = binomial(m, 2)
    dtau = kendall_tau_dict
    # Compute the average normalized distance:
    avg_dist = sum(dtau(r, consensus_j) for r in profile_i) / (n_i * norm_factor)
    return avg_dist
end


"""
    overall_divergence(group_profiles, consensus_map)

Given:
  - group_profiles: Dict{T,Vector{Dict{Symbol,Int}}} mapping group id to its profile.
  - consensus_map: Dict{T,Dict{Symbol,Int}} mapping group id to its consensus ranking.
  
Assumes each ranking involves m alternatives (inferred from any consensus ranking).
Computes the divergence measure

   D = (1/((k-1))) * sum_{i ≠ j} ( (n_i/n) * AvgDist(G_i, ρ_j) )

where AvgDist(G_i, ρ_j) is computed using pairwise_divergence.
"""
function overall_divergence(group_profiles,
                            consensus_map)
    groups = keys(group_profiles)
    k = length(groups)
    # total number of rankings across all groups
    n = sum(length(profile) for profile in values(group_profiles))
    # infer m from any consensus ranking (assumes all have same number of alternatives)
    m = length(first(values(consensus_map)))
    
    total = 0.0
    for i in groups
        n_i = length(group_profiles[i])
        for j in groups
            if i != j
                # divergence from group i to consensus ranking of group j
                d_ij = pairwise_group_divergence(group_profiles[i], consensus_map[j], m)
               # println("Divergence from group $i to consensus of group $j: $d_ij")
               
               #println("n_i: $n_i, n: $n")
                total += (n_i / n) * d_ij
            end
        end
    end
    #println(total)
    D = total / ((k - 1))
    return D
end

"""
    overall_divergences(grouped_consensus, whole_df, key)

Convenience wrapper that constructs the inputs required by
[`overall_divergence`] from grouped consensus data and the full dataframe.
"""
function overall_divergences(grouped_consensus, whole_df, key)
    k = nrow(grouped_consensus)
    groups_profiles = Dict(grouped_consensus[i,key] =>
    map(x->x.profile, Base.filter(x-> x[key] == grouped_consensus[i,key], eachrow(whole_df)))
    for i in 1:k)
    consensus_map = Dict(i[key] => i.x1 for i in eachrow(grouped_consensus))

    D = overall_divergence(groups_profiles, consensus_map)

return D
end


"""
    apply_measure_to_bts(bts, measure)

Apply a polarization measure to each bootstrapped dataset in `bts`.
"""
function apply_measure_to_bts(bts, measure)
    Dict(x => map(y->measure(y.profile), bts[x]) for x in keys(bts))
end


"""
    apply_all_measures_to_bts(bts; measures)

Apply multiple polarization measures to the bootstrapped datasets in `bts`.
Returns a dictionary keyed by measure name.
"""
function apply_all_measures_to_bts(bts; measures = [Ψ,  calc_total_reversal_component,
                     calc_reversal_HHI, fast_reversal_geometric])
        Dict(nameof(measure) => apply_measure_to_bts(bts, measure) for measure in measures)
end


"""
    compute_group_metrics(df, demo)

Compute coherence and divergence metrics for the demographic variable `demo`
in dataframe `df`.
"""
function compute_group_metrics(df::DataFrame, demo)
    g = groupby(df, demo)
    results_distance = combine(g) do subdf
        group_avg_distance(subdf)
    end

    prop = proportionmap(df[!, demo])
    C = weighted_coherence(results_distance, prop, demo)

    consensus = combine(g) do subdf
        consensus_for_group(subdf)
    end


    D = overall_divergences(consensus, df, demo)

    return C, D
end


"""
    bootstrap_group_metrics(bt_profiles, demo)

Run `compute_group_metrics` across many bootstrap replications.
Returns a nested dictionary mapping each variant to vectors of coherence (`C`)
and divergence (`D`) values.
"""
function bootstrap_group_metrics(bt_profiles, demo)
    result = Dict{Symbol, Dict{Symbol, Vector{Float64}}}()

    for (variant, reps) in bt_profiles
        Cvals = Float64[]
        Dvals = Float64[]

        for df in reps
            C, D = compute_group_metrics(df, demo)
            push!(Cvals, C)
            push!(Dvals, D)
        end

        result[variant] = Dict(:C => Cvals,
                               :D => Dvals)
    end

    return result
end
