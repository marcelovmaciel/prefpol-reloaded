
# === Project and Data Setup ===

"""
    load_spss_file(path::String)

Read an SPSS `.sav` file using R's `haven` package and return a `DataFrame`.
"""
function load_spss_file(path::String)
    RCall.reval("library(haven)")
    RCall.reval("library(mice)")

    @rput path  # this macro works fine for variables
    return rcopy(R"read_sav(path)")
end


# === Score Processing ===
"""
    build_candidate_score_distributions(df, candidatos)

Create a countmap of scores for each candidate column in `df`.
"""
function build_candidate_score_distributions(df::DataFrame, candidatos::Vector{String})
    return Dict(c => countmap(df[!, c]) for c in candidatos)
end


"""
    convert_keys_to_int(dict)

Return a new dictionary with keys converted to integers. Float keys that
are approximately integers are rounded; others throw an error.
"""
function convert_keys_to_int(dict)
    return Dict(
        if k isa Integer
            k
        elseif k isa AbstractFloat
            r = round(Int, k)
            if !isapprox(k, r; atol=1e-8)
                throw(ArgumentError("Expected near-integer key, got $k"))
            end
            r
        else
            throw(ArgumentError("Unsupported key type: $(typeof(k))"))
        end => v
        for (k, v) in dict
    )
end


"""
    sanitize_countmaps(countmaps)

Convert the keys of each countmap to integers.
"""
function sanitize_countmaps(countmaps::Dict{String,<:AbstractDict})
    return Dict(c => convert_keys_to_int(cm) for (c, cm) in countmaps)
end


"""
    compute_dont_know_her(countmaps, nrespondents)

Compute the percentage of "don't know" codes (96–99) for each candidate and
return a sorted vector of `(name, percent)` pairs.
"""
function compute_dont_know_her(countmaps::Dict{String,Dict{Int,Int}}, nrespondents::Int)
    return sort([
        (k, 100 * sum(get(v, code, 0) for code in (96, 97, 98,99)) / nrespondents)
        for (k, v) in countmaps
    ], by = x -> x[2])
end



"""
    prepare_scores_for_imputation_int(df, score_cols; extra_cols=String[])

Convert numeric score columns to `Int` and mark special codes (96–99) as
missing. Optional `extra_cols` are appended unchanged.
"""
function prepare_scores_for_imputation_int(df::DataFrame,
    score_cols::Vector{String};
    extra_cols::Vector{String}=String[])
    # (1) Split truly numeric from anything that isn't
    numeric_cols   = Base.filter(c -> eltype(df[!, c]) <: Union{Missing, Real}, score_cols)
    nonnumeric     = setdiff(score_cols, numeric_cols)
    if !isempty(nonnumeric)
        @warn "prepare_scores_for_imputation_int: skipping non‑numeric columns $(nonnumeric)"
    end

    # (2) Work only on the numeric score columns
    scores_int = mapcols(col -> Int.(col), df[:, numeric_cols])
    declared   = Impute.declaremissings(scores_int; values = (96, 97, 98, 99))

    # (3) Append any extra (demographic) columns, untouched
    return isempty(extra_cols) ? declared : hcat(declared, df[:, extra_cols])
end


"""
    prepare_scores_for_imputation_categorical(df, score_cols; extra_cols = String[])

Same idea as the `_int` version but returns the scores as **ordered categoricals**.
Demographics are still appended unchanged.
"""
function prepare_scores_for_imputation_categorical(df::DataFrame,
                                                   score_cols::Vector{String};
                                                   extra_cols::Vector{String}=String[])
    declared = prepare_scores_for_imputation_int(df, score_cols; extra_cols = String[])
    declared_cat = mapcols(col -> categorical(col, ordered = true), declared)
    return isempty(extra_cols) ? declared_cat : hcat(declared_cat, df[:, extra_cols])
end


# === Slice Top  ===
"""
    get_most_known_candidates(dont_know_her, how_many)

Return the names of the `how_many` candidates with the lowest "don't know"
percentages from a precomputed list.
"""
function get_most_known_candidates(dont_know_her::Vector{Tuple{String, Float64}}, how_many)
    most_known_candidates = [x[1] for x in dont_know_her[1:how_many]]    
    return most_known_candidates
end


"""
    select_top_candidates(countmaps, nrespondents; m, force_include=String[])

Select up to `m` candidates with the lowest "don't know" rates. Additional
names in `force_include` are guaranteed to appear in the result.
"""
function select_top_candidates(countmaps::Dict{String,<:AbstractDict},
                               nrespondents::Int;
                               m::Int,
                               force_include::Vector{String}=String[])

    # remove duplicates, keep order
    inc = unique(force_include)

    # truncate if too many
    if length(inc) > m
        @warn "force_include has more than $m names; truncating to first $m."
        inc = inc[1:m]
    end

    # ---------- popularity list, already sorted ascending by “don't-know-her” ----
    poplist = [name
               for (name, _) in compute_dont_know_her(countmaps, nrespondents)
               if name ∉ inc]                         # drop forced names, duplicates

    needed  = m - length(inc)
    extra   = needed > 0 ? poplist[1:min(needed, length(poplist))] : String[]

    selected = vcat(inc, extra)

    if length(selected) < m
        @warn "Only $(length(selected)) unique candidates available; requested $m."
    end

    return selected
end


"""
    compute_candidate_set(scores_df; candidate_cols, m, force_include=String[])

Determine the set of `m` candidates to analyze based on score distributions.
"""
function compute_candidate_set(scores_df::DataFrame;
                               candidate_cols,
                               m::Int,
                               force_include::Vector{String} = String[])

    countmaps     = build_candidate_score_distributions(scores_df, candidate_cols)
    countmaps2    = sanitize_countmaps(countmaps)
    nrespondents  = nrow(scores_df)

    return select_top_candidates(countmaps2, nrespondents;
                                 m = m, force_include = force_include)
end



"""
    get_df_just_top_candidates(df, how_top; demographics=String[])
    get_df_just_top_candidates(df, which_ones; demographics=String[])

Return `df` restricted to selected candidate columns plus optional
`demographics`. `how_top` chooses the top `how_top` candidates from the
global `dont_know_her` list, while `which_ones` specifies candidate names
directly.
"""
function get_df_just_top_candidates(df::DataFrame, how_top::Int; demographics = String[] )
    most_known_candidates = get_most_known_candidates(dont_know_her, how_top)
    return df[!, vcat(most_known_candidates, demographics)]
end

function get_df_just_top_candidates(df::DataFrame, which_ones; demographics = String[])
    return df[!, vcat(which_ones, demographics)]
end


"""
    GLOBAL_R_IMPUTATION(df; m=1)

Impute missing values using R's `mice` package and return a completed
DataFrame. A random seed is set for reproducibility across bootstraps.
"""
const GLOBAL_R_IMPUTATION = let
    function f(df::DataFrame; m::Int = 1)
        # random seed to keep bootstrap independence
        seed = rand(1:10^6)
        RCall.reval("set.seed($seed)")

        R"""
        suppressPackageStartupMessages(library(mice))

        df <- as.data.frame($df)

        # ---------- boilerplate ----------
        init <- mice(df, maxit = 0, print = FALSE)
        meth <- init$method                   # default methods
        pred <- make.predictorMatrix(df)
        diag(pred) <- 0                       # no self-prediction

        # ---------- customise methods ----------
        for (v in names(df)) {
          col <- df[[v]]
          if (all(is.na(col)) || length(unique(na.omit(col))) <= 1) {
            meth[v] <- ""                     # constant or all-missing
          } else if (is.factor(col)) {
            n_cat <- nlevels(col)
            if (n_cat == 2) {
              meth[v] <- "logreg"             # binomial GLM
            } else {                          # 3+ categories
              meth[v] <- "cart"               # safe, no weight explosion
            }
          } else if (is.numeric(col)) {
            meth[v] <- "pmm"
          }
        }

        # ---------- one imputation ----------
        imp <- mice(df,
                    m               = $m,
                    method          = meth,
                    predictorMatrix = pred,
                    printFlag       = FALSE)

        completed_df <- complete(imp, 1)
        """
        return rcopy(DataFrame, R"completed_df")
    end
end

# === End of Slice Top ===
# === Imputation ===


"""
    imputation_variants(df, candidates, demographics; most_known_candidates=String[])

Return a named tuple of imputed score tables using several strategies:
zero replacement, random sampling, and R's `mice` package.
"""
function imputation_variants(df::DataFrame,
    candidates::Vector{String},
    demographics::Vector{String};
    most_known_candidates::Vector{String}=String[])

# 1 ─ Determine which score columns to use
use_cols = isempty(most_known_candidates) ? candidates : most_known_candidates

# 2 ─ Subset to relevant columns (if top-candidates requested)
df_subset = isempty(most_known_candidates) ? df : get_df_just_top_candidates(df, use_cols; demographics = demographics)

# 3 ─ Prepare imputation tables (same for both branches now)
scores_int  = prepare_scores_for_imputation_int(df_subset, use_cols; extra_cols = demographics)
scores_cat  = prepare_scores_for_imputation_categorical(df_subset, use_cols; extra_cols = demographics)

# 4 ─ Apply imputation variants
imputed0    = Impute.replace(scores_int, values = 0)
imputedRnd  = Impute.impute(scores_cat, Impute.SRS(; rng = MersenneTwister()))
imputedM    = GLOBAL_R_IMPUTATION(scores_cat)

return (zero = imputed0,
random = imputedRnd,
mice = imputedM)
end

# === End of Imputation ===


"""
    weighted_bootstrap(data, weights, B)

Draw `B` bootstrap resamples from `data` using the provided `weights`.
"""
function weighted_bootstrap(data::DataFrame, weights::Vector{Float64}, B::Int)
    n = nrow(data)
    boot_samples = Vector{DataFrame}(undef, B)
    
    for b in 1:B
        idxs = sample(1:n, Weights(weights), n; replace=true)
        boot_samples[b] = data[idxs, :]
    end
    
    return boot_samples
end



"""
    get_row_candidate_score_pairs(row, score_cols)

Create a dictionary mapping candidate names to their scores in `row`.
"""
function get_row_candidate_score_pairs(row, score_cols)
    Dict(Symbol(c) => row[c] for c in score_cols)
end

"""
    get_order_dict(score_dict)

Convert score values to ranks where higher scores receive smaller rank numbers.
"""
function  get_order_dict(score_dict)
    unique_scores = sort(unique(values(score_dict)); rev = true)
    lookup = Dict(s => r for (r,s) in enumerate(unique_scores))
    Dict(k => lookup[v] for (k,v) in score_dict)
end

"""
    force_scores_become_linear_rankings(score_dict; rng=MersenneTwister())

Break ties in `score_dict` at random to obtain a linear ranking.
"""
function force_scores_become_linear_rankings(score_dict; rng=MersenneTwister())

    grouped = Dict(score => Symbol[] for score in unique(values(score_dict)))
    
    for (cand, score) in score_dict
        push!(grouped[score], cand)
    end

    sorted_scores = sort(collect(keys(grouped)), rev=true)
    linear_ranking = Dict{Symbol, Int}()
    next_rank = 1

    for score in sorted_scores
        cands = grouped[score]
        shuffle!(rng, cands)
        for cand in cands
            linear_ranking[cand] = next_rank
            next_rank += 1
        end
    end

    return linear_ranking
end




"""
    build_profile(df; score_cols, rng=Random.GLOBAL_RNG, kind=:linear)

Construct ranking dictionaries for each row of `df`. `kind` may be
`:linear` to break ties randomly or `:weak` to keep weak orderings.
"""
function build_profile(df::DataFrame;
                       score_cols::Vector,
                       rng  = Random.GLOBAL_RNG,
                       kind::Symbol = :linear)   # :linear or :weak
    f = kind === :linear ? force_scores_become_linear_rankings : get_order_dict
    score_dicts = map(row -> get_row_candidate_score_pairs(row, score_cols),
                      eachrow(df))
    
    return map(sd -> f(sd), score_dicts)
end


"""
    profile_dataframe(df; score_cols, demo_cols, rng=Random.GLOBAL_RNG, kind=:linear)

Return a DataFrame with a `:profile` column of rankings and the requested
demographic columns.
"""
function profile_dataframe(df::DataFrame;
                           score_cols::Vector,
                           demo_cols::Vector,
                           rng  = Random.GLOBAL_RNG,
                           kind::Symbol = :linear)
    prof = build_profile(df; score_cols = score_cols, rng = rng, kind = kind)
    return DataFrame(profile = prof) |> (d -> hcat(d, df[:, demo_cols]))
end



"""
    dict2svec(d; cs=cands, higher_is_better=false)

Encode a dictionary of candidate scores into a static vector permutation.
"""
@inline function dict2svec(d::Dict{Symbol,<:Integer}; cs::Vector{Symbol}=cands,
                           higher_is_better::Bool=false)
    # 1. pack the m scores into an isbits StaticVector
    m = length(cs)                                # number of candidates
    vals = SVector{m,Int}(map(c -> d[c], cs))

    # 2. permutation that sorts those scores
    perm = sortperm(vals; rev = higher_is_better)           # Vector{Int}

    # 3. return as SVector{m,UInt8} (10 B if m ≤ 10)
    return SVector{m,UInt8}(perm)
end




"""
    decode_rank(code, pool)

Return the ranking `SVector` for a pooled integer `code`. If `code` is
already an `SVector`, it is returned unchanged.
"""
decode_rank(code::Integer,      pool) = pool[code]  # original pool lookup behaviour
decode_rank(r::SVector, _) = r                      # no-op if already SVector


"""
    compress_rank_column!(df, cands; col=:profile)

Encode ranking dictionaries in column `col` into pooled static vectors for
memory efficiency. Returns the pool used for decoding.
"""
function compress_rank_column!(df::DataFrame, cands; col::Symbol=:profile)
    # 1. Dict → SVector
    
    sv = [dict2svec(r[col],cs = cands) for r in eachrow(df)]  # one tiny allocation per row

    # 2. pool identical SVectors (UInt16 index)
    pooled = PooledArray(sv; compress = true)

    # 3. overwrite in-place; let Dict objects be GC’d
    df[!, col] = pooled
    metadata!(df, "candidates", Symbol.(cands)) # new, test
    GC.gc()                       # reclaim Dict storage promptly
    return pooled.pool            # decoder lookup table
end



"""
    perm2dict(perm, cs)

Translate a permutation of candidate indices into a dictionary mapping
candidates to their ranks.
"""
@inline function perm2dict(perm::AbstractVector{<:Integer},
                           cs::Vector{Symbol})
    d = Dict{Symbol,Int}()
    @inbounds for (place, idx) in pairs(perm)          # place = 1,2,…
        d[cs[idx]] = place
    end
    return d
end


perm_to_dict = @inline perm2dict


"""
    decode_profile_column!(df)

Decode a compressed `:profile` column back into dictionaries using candidate
metadata stored in the DataFrame.
"""
function decode_profile_column!(df::DataFrame)
    eltype(df.profile) <: Dict && return df            # nothing to do

    cand_syms = metadata(df, "candidates")
    col       = df.profile
    decoded   = Vector{Dict{Symbol,Int}}(undef, length(col))

    if col isa PooledArray
        pool = col.pool
        for j in eachindex(col)
            perm = decode_rank(col[j], pool)
            decoded[j] = perm_to_dict(perm, cand_syms)
        end
    else                                               # plain Vector{SVector}
        for j in eachindex(col)
            decoded[j] = perm_to_dict(col[j], cand_syms)
        end
    end

    df[!, :profile] = decoded
    return df
end


"""
    decode_each!(var_map)

Decode the profile column of each DataFrame stored in `var_map`.
"""
@inline function decode_each!(var_map)
    for vec in values(var_map)          # vec::Vector{DataFrame}
        decode_profile_column!(vec[1])  # length == 1 in streaming path
    end
end
