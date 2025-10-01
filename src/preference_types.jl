# src/preferences_types.jl — Minimal surface (WeakRank/StrictRank/Pairwise)
using StaticArrays: SVector, SMatrix

const PreferenceIndex = UInt16
const CandidateId     = PreferenceIndex  # alias when referring to ids

# ============================ CandidatePool =============================

"""
    CandidatePool{N}

Immutable mapping between candidate names (Symbols) and integer ids `1..N`.

- `names::SVector{N,Symbol}`: canonical ordered list of candidates
- `index::Dict{Symbol,CandidateId}`: name → id

Construction:
- `CandidatePool(::AbstractVector{Symbol})`
- `CandidatePool(::AbstractVector{<:AbstractString})`

Conveniences:
- `pool[sym]::CandidateId` (name→id)
- `pool[i::Integer]::Symbol` (id→name)
- `candidates(pool)::Vector{Symbol}`
- `to_cmap(pool)::Dict{CandidateId,Symbol}`
"""
struct CandidatePool{N}
    names::SVector{N,Symbol}
    index::Dict{Symbol,CandidateId}
end

function CandidatePool(names::AbstractVector{Symbol})
    cand_tuple = Tuple(names)
    N = length(cand_tuple)
    N == 0 && throw(ArgumentError("Candidate pool cannot be empty"))
    length(Set(cand_tuple)) == N || throw(ArgumentError("Candidates must be unique"))
    name_vec = SVector{N,Symbol}(cand_tuple)
    idx = Dict{Symbol,CandidateId}(s => CandidateId(i) for (i, s) in enumerate(name_vec))
    return CandidatePool{N}(name_vec, idx)
end
CandidatePool(names::AbstractVector{<:AbstractString}) = CandidatePool(Symbol.(names))

Base.length(pool::CandidatePool{N}) where {N} = N
function Base.iterate(pool::CandidatePool{N}, state::Int=1) where {N}
    state > N && return nothing
    return pool.names[state], state + 1
end
Base.getindex(pool::CandidatePool, cand::Symbol) = pool.index[cand]
Base.getindex(pool::CandidatePool{N}, idx::Integer) where {N} = pool.names[idx]

"""
    candidateat(pool, idx) -> Symbol

Alias for `pool[idx]`.
"""
candidateat(pool::CandidatePool{N}, idx::Integer) where {N} = pool.names[idx]

"""
    keys(pool)

Iterate candidate names in canonical order (returns the SVector of names).
"""
Base.keys(pool::CandidatePool) = pool.names

"""
    candidates(pool) -> Vector{Symbol}

Materialize candidate names as a `Vector`.
"""
candidates(pool::CandidatePool{N}) where {N} = collect(pool.names)

"""
    to_cmap(pool) -> Dict{CandidateId,Symbol}

Build an ID→name map compatible with common “cmap” conventions.
"""
function to_cmap(pool::CandidatePool{N}) where {N}
    d = Dict{CandidateId,Symbol}()
    @inbounds for i in 1:N
        d[CandidateId(i)] = pool.names[i]
    end
    return d
end

# ============================ Preference types ===========================

abstract type AbstractPreference{N} end

"""
    WeakRank{N} <: AbstractPreference{N}

Rank-encoded ballot allowing ties and truncation.
- `ranks::SVector{N,Union{Missing,PreferenceIndex}}`
  * lower is better; equal values indicate indifference; `missing` = unranked
"""
struct WeakRank{N} <: AbstractPreference{N}
    ranks::SVector{N,Union{Missing,PreferenceIndex}}
end

"""
    StrictRank{N} <: AbstractPreference{N}

Strict total order encoded as contiguous ranks `1..N` with no `missing` and no ties.
- `ranks::SVector{N,PreferenceIndex}`
"""
struct StrictRank{N} <: AbstractPreference{N}
    ranks::SVector{N,PreferenceIndex}
    function StrictRank{N}(r::SVector{N,PreferenceIndex}) where {N}
        # contiguous 1..N and unique
        length(Set(r)) == N || throw(ArgumentError("Strict ranks must be unique"))
        minimum(Int.(r)) == 1 || throw(ArgumentError("Strict ranks must start at 1"))
        maximum(Int.(r)) == N || throw(ArgumentError("Strict ranks must be 1..N"))
        return new{N}(r)
    end
end

"""
    Pairwise{N,T<:Integer} <: AbstractPreference{N}

Ballot-level pairwise comparison sign matrix.
Entry meanings: `+1` if `i ≻ j`, `-1` if `j ≻ i`, `0` if tie/no-comparison.
"""
struct Pairwise{N,T<:Integer} <: AbstractPreference{N}
    matrix::SMatrix{N,N,T}
end

# ============================ WeakOrder view =============================

"""
    WeakOrder{N}

Indifference classes: `levels[1]` is the top tie-group, etc. Every candidate 1..N
must appear exactly once. Unranked (from WeakRank) are placed in the *last* level.
"""
struct WeakOrder{N}
    levels::Vector{Vector{CandidateId}}
    function WeakOrder{N}(levels::Vector{Vector{CandidateId}}) where {N}
        seen = Set{CandidateId}()
        for lvl in levels
            for c in lvl
                (c < 1 || c > CandidateId(N)) && throw(ArgumentError("Candidate id $(c) out of 1..$N"))
                in(c, seen) && throw(ArgumentError("Candidate id $(c) appears in multiple levels"))
                push!(seen, c)
            end
        end
        length(seen) == N || throw(ArgumentError("WeakOrder must cover all $N candidates exactly once"))
        return new{N}(levels)
    end
end
Base.length(::WeakOrder{N}) where {N} = N
Base.iterate(wo::WeakOrder{N}, state::Int=1) where {N} = state > N ? nothing : (wo.levels[state], state+1)
function Base.show(io::IO, wo::WeakOrder{N}) where {N}
    print(io, "WeakOrder{$N}(", wo.levels, ")")
end

"""
    weakorder_symbol_groups(wo, pool) -> Vector{Vector{Symbol}}

Render levels of a `WeakOrder` using candidate names.
"""
function weakorder_symbol_groups(wo::WeakOrder{N}, pool::CandidatePool{N}) where {N}
    [map(i -> pool.names[Int(i)], lvl) for lvl in wo.levels]
end

# ============================ Helpers ====================================

@inline _nonmissing(xs) = (x for x in xs if !ismissing(x))

"""
    _to_prefindex_or_missing(x) -> Union{Missing,PreferenceIndex}

Convert an input to `PreferenceIndex` (rejecting non-positive). Pass `missing` through.
"""
@inline function _to_prefindex_or_missing(x)
    ismissing(x) && return missing
    xi = Int(x)
    xi < 1 && throw(ArgumentError("Ranks must be ≥ 1, got $xi"))
    return PreferenceIndex(xi)
end

"""
    _collect_ranks(pool, dict) -> NTuple{N,Union{Missing,PreferenceIndex}}

Collect ranks from a `{name => rank}` dict for all candidates in `pool`.
"""
@inline function _collect_ranks(pool::CandidatePool{N}, dict::AbstractDict{Symbol,<:Integer}) where {N}
    ntuple(i -> begin
        s = pool.names[i]
        if haskey(dict, s)
            _to_prefindex_or_missing(dict[s])
        else
            missing
        end
    end, Val(N))
end

# ============================ Constructors ===============================

"""
    WeakRank(pool, dict)
    WeakRank(pool, vec)

Build a `WeakRank{N}` from a dict `{name=>rank}` (missing keys become `missing`),
or a vector of length `N` containing integers or `missing`.
"""
function WeakRank(pool::CandidatePool{N}, dict::AbstractDict{Symbol,<:Integer}) where {N}
    ranks = _collect_ranks(pool, dict)
    return WeakRank{N}(SVector{N,Union{Missing,PreferenceIndex}}(ranks))
end
function WeakRank(pool::CandidatePool{N}, v::AbstractVector{T}) where {N,T}
    length(v) == N || throw(ArgumentError("Expected $(N) ranks, got $(length(v))"))
    ranks = ntuple(i -> _to_prefindex_or_missing(v[i]), Val(N))
    return WeakRank{N}(SVector{N,Union{Missing,PreferenceIndex}}(ranks))
end

"""
    StrictRank(pool, v::AbstractVector{<:Integer})

Build a `StrictRank{N}` from a dense vector of length `N` (must be a permutation 1..N).
"""
function StrictRank(pool::CandidatePool{N}, v::AbstractVector{<:Integer}) where {N}
    length(v) == N || throw(ArgumentError("Expected $(N) ranks, got $(length(v))"))
    tu = ntuple(i -> PreferenceIndex(v[i]), Val(N))
    return StrictRank{N}(SVector{N,PreferenceIndex}(tu))
end


"""
    StrictRank(pool, order::AbstractVector{<:Symbol})

Build a `StrictRank{N}` from a full permutation of candidate names in `pool`.
Throws if any name is unknown, duplicated, or if the permutation length ≠ `N`.
"""
function StrictRank(pool::CandidatePool{N}, order::AbstractVector{<:Symbol}) where {N}
    length(order) == N || throw(ArgumentError("Expected $(N) candidates, got $(length(order))"))

    # map names → ids while checking for unknowns/dupes
    seen = Set{Symbol}()
    ids  = Vector{Int}(undef, N)
    @inbounds for (pos, s) in enumerate(order)
        haskey(pool.index, s) || throw(ArgumentError("Unknown candidate name: $(s)"))
        in(s, seen) && throw(ArgumentError("Duplicate candidate in permutation: $(s)"))
        push!(seen, s)
        ids[pos] = Int(pool[s])
    end
    # ensure the permutation covers all candidates exactly once
    length(seen) == N || throw(ArgumentError("Permutation must include all $(N) candidates exactly once"))

    # build ranks: for candidate id c, rank is its position in `order`
    ranks = Vector{PreferenceIndex}(undef, N)
    @inbounds for (pos, cid_int) in enumerate(ids)
        ranks[cid_int] = PreferenceIndex(pos)
    end
    return StrictRank{N}(SVector{N,PreferenceIndex}(Tuple(ranks)))
end





# ============================ Views / Conversions =========================

"""
    to_weak(x::StrictRank) -> WeakRank

Embed a strict order as `WeakRank` (no `missing`, no ties introduced).
"""
function to_weak(x::StrictRank{N}) where {N}
    return WeakRank{N}(SVector{N,Union{Missing,PreferenceIndex}}(ntuple(i->x.ranks[i], Val(N))))
end

"""
    to_strict(x::WeakRank; tie_break=:error | :linearize | f) -> StrictRank

Convert a possibly tied/truncated rank into a strict total order.
- `:error` → throw if any `missing` or ties
- `:linearize` → order by `(rank, id)`, `missing` last, then map to `1..N`
- custom `f(i,j,ranks,pool)` used only to break ties (when ranks equal); `missing` still last
"""
function to_strict(x::WeakRank{N}; tie_break=:linearize, pool::Union{Nothing,CandidatePool{N}}=nothing) where {N}
    r = x.ranks
    if tie_break === :error
        all(!ismissing, r) || throw(ArgumentError("Cannot strictify: missing ranks present"))
        vals = collect(r)
        length(Set(vals)) == N || throw(ArgumentError("Cannot strictify: ties present"))
        tu = ntuple(i->PreferenceIndex(Int(vals[i])), Val(N))
        return StrictRank{N}(SVector{N,PreferenceIndex}(tu))
    end
    # linearize or custom comparator
    order = collect(1:N)
    if tie_break === :linearize
        sort!(order, by = i -> (ismissing(r[i]) ? (true, typemax(Int), i) : (false, Int(r[i]), i)))
    elseif tie_break isa Function
        # Sort by rank, then break ties with user comparator f(i,j,ranks,pool)
        f = tie_break
        sort!(order, lt = (i,j)->begin
            ai, aj = r[i], r[j]
            if ismissing(ai) && ismissing(aj)
                return i < j
            elseif ismissing(ai)
                return false
            elseif ismissing(aj)
                return true
            elseif ai == aj
                return f(i,j,r,pool)
            else
                return Int(ai) < Int(aj)
            end
        end)
    else
        throw(ArgumentError("Unknown tie_break=$(tie_break). Use :error, :linearize, or a function."))
    end
    ranks = Vector{PreferenceIndex}(undef, N)
    @inbounds for (pos, idx) in enumerate(order)
        ranks[idx] = PreferenceIndex(pos)
    end
    return StrictRank{N}(SVector{N,PreferenceIndex}(Tuple(ranks)))
end

"""
    to_perm(x) -> SVector{N,PreferenceIndex}

Return the permutation (candidate-ids) ordered from best to worst according to `x`.
For `WeakRank`, this is the linearization order `(rank,id)` with `missing` last.
"""
function to_perm(x::StrictRank{N}) where {N}
    # inverse permutation: position → candidate id
    inv = Vector{PreferenceIndex}(undef, N)
    @inbounds for (pos, idx) in enumerate(sortperm(collect(1:N), by=i->Int(x.ranks[i])))
        inv[pos] = PreferenceIndex(idx)
    end
    return SVector{N,PreferenceIndex}(Tuple(inv))
end
function to_perm(x::WeakRank{N}) where {N}
    r = x.ranks
    order = sort(collect(1:N), by = i -> (ismissing(r[i]) ? (true, typemax(Int), i) : (false, Int(r[i]), i)))
    return SVector{N,PreferenceIndex}(Tuple(PreferenceIndex(i) for i in order))
end

"""
    to_weakorder(x) -> WeakOrder

Convert ranks to explicit indifference classes (levels). Unranked become the last level.
"""
function to_weakorder(x::WeakRank{N}) where {N}
    groups = Dict{Int,Vector{CandidateId}}()
    unranked = CandidateId[]
    @inbounds for i in 1:N
        ri = x.ranks[i]
        if ismissing(ri)
            push!(unranked, CandidateId(i))
        else
            push!(get!(groups, Int(ri), CandidateId[]), CandidateId(i))
        end
    end
    ranked_levels = [groups[k] for k in sort!(collect(keys(groups)))]
    levels = isempty(unranked) ? ranked_levels : vcat(ranked_levels, [unranked])
    return WeakOrder{N}(levels)
end
function to_weakorder(x::StrictRank{N}) where {N}
    # each level is singleton in increasing rank
    id_by_rank = Vector{CandidateId}(undef, N)
    # build mapping candidate id by position
    perm = sortperm(collect(1:N), by = i -> Int(x.ranks[i]))
    for (pos, cid) in enumerate(perm)
        id_by_rank[pos] = CandidateId(cid)
    end
    return WeakOrder{N}([[id_by_rank[k]] for k in 1:N])
end


"""
    StrictRankView(r, pool)

Lightweight pretty-print wrapper that renders a `StrictRank` using candidate names
instead of numeric indices. Use via `pretty(r, pool)`.
"""
struct StrictRankView{N}
    r::StrictRank{N}
    pool::CandidatePool{N}
end

"""
    pretty(r::StrictRank, pool::CandidatePool) -> StrictRankView

Return a wrapper that customizes `show` to print candidate names in rank order.
"""
pretty(r::StrictRank{N}, pool::CandidatePool{N}) where {N} = StrictRankView{N}(r, pool)

function Base.show(io::IO, v::StrictRankView{N}) where {N}
    names = ordered_candidates(v.r, v.pool)
    print(io, "StrictRank(", join(string.(names), " ≻ "), ")")
end

"""
    WeakOrderView(wo, pool; hide_unranked = false)

Pretty-print wrapper for `WeakOrder` using candidate names.
Within a level, names are joined by `~`; between levels by `≻`.
If `hide_unranked=true`, the last level is treated as the unranked bucket and
is omitted from the display, with a note listing omitted names.
(Assumes `to_weakorder(::WeakRank)` places unranked last.)
"""
struct WeakOrderView{N}
    wo::WeakOrder{N}
    pool::CandidatePool{N}
    hide_unranked::Bool
end

"""
    pretty(wo::WeakOrder, pool; hide_unranked=false) -> WeakOrderView

Return a view wrapper that pretty-prints with names and separators.
"""
pretty(wo::WeakOrder{N}, pool::CandidatePool{N}; hide_unranked::Bool=false) where {N} =
    WeakOrderView{N}(wo, pool, hide_unranked)

function Base.show(io::IO, v::WeakOrderView{N}) where {N}
    levels = v.wo.levels
    omitted = Symbol[]

    # Optionally treat the last level as "unranked" and omit it from the main display.
    if v.hide_unranked && !isempty(levels)
        un = levels[end]
        if !isempty(un)
            append!(omitted, [v.pool.names[Int(i)] for i in un])
        end
        levels = levels[1:end-1]
    end

    # Build human-readable pieces: "~" within a level, "≻" between levels.
    level_strs = String[]
    for ids in levels
        syms = [v.pool.names[Int(i)] for i in ids]
        push!(level_strs, join(string.(syms), " ~ "))
    end

    print(io, "WeakOrder(", join(level_strs, " ≻ "), ")")
    if !isempty(omitted)
        print(io, "  (unranked omitted: ", join(string.(omitted), ", "), ")")
    end
end



# ============================ Pairwise ===================================

"""
    ext_bottom(ra, rb, i, j, ranks, pool) -> Int

Default extension: ranked ≻ unranked. Both unranked ⇒ 0.
"""
@inline function ext_bottom(ra, rb, i, j, ranks, pool)
    if ismissing(ra) && ismissing(rb)
        return 0
    elseif ismissing(ra)
        return -1
    elseif ismissing(rb)
        return 1
    else
        d = Int(rb) - Int(ra)
        return d > 0 ? 1 : d < 0 ? -1 : 0
    end
end

"""
    ext_none(ra, rb, i, j, ranks, pool) -> Int

Strict-only comparisons: any `missing` ⇒ 0.
"""
@inline function ext_none(ra, rb, i, j, ranks, pool)
    if ismissing(ra) || ismissing(rb)
        return 0
    else
        d = Int(rb) - Int(ra)
        return d > 0 ? 1 : d < 0 ? -1 : 0
    end
end

@inline function _resolve_extension(extension)
    if extension === :bottom || extension === nothing
        return ext_bottom
    elseif extension === :none
        return ext_none
    elseif extension isa Function
        return extension
    else
        throw(ArgumentError("Unknown extension $(extension). Use :bottom, :none, or a function."))
    end
end

@inline function _pairwise_entry(ranks::SVector{N,Union{Missing,PreferenceIndex}}, i::Int, j::Int, ext, pool) where {N}
    i == j && return Int8(0)
    ai = ranks[i]; bj = ranks[j]
    return Int8(ext(ai, bj, i, j, ranks, pool))
end

"""
    to_pairwise(x, pool; extension=:bottom) -> Pairwise{N,Int8}

Build ballot-level pairwise sign matrix.
"""
function to_pairwise(x::WeakRank{N}, pool::CandidatePool{N}; extension=:bottom) where {N}
    ext = _resolve_extension(extension)
    data = ntuple(i -> ntuple(j -> _pairwise_entry(x.ranks, i, j, ext, pool), Val(N)), Val(N))
    return Pairwise{N,Int8}(SMatrix{N,N,Int8}(data))
end
function to_pairwise(x::StrictRank{N}, pool::CandidatePool{N}; extension=:bottom) where {N}
    return to_pairwise(to_weak(x), pool; extension=extension)
end

# ============================ Predicates & utils ==========================

"""
    rank(x, pool, name::Symbol) -> Union{Missing,PreferenceIndex}

Lookup a candidate’s rank in a ballot (`missing` possible for `WeakRank`).
"""
function rank(x::WeakRank{N}, pool::CandidatePool{N}, cand::Symbol) where {N}
    return x.ranks[pool[cand]]
end
function rank(x::StrictRank{N}, pool::CandidatePool{N}, cand::Symbol) where {N}
    return x.ranks[pool[cand]]
end

"""
    prefers(x, pool, a::Symbol, b::Symbol) -> Bool

True iff both ranked and `a` strictly better than `b`.
"""
function prefers(x::WeakRank, pool::CandidatePool, a::Symbol, b::Symbol)
    ra = rank(x, pool, a); rb = rank(x, pool, b)
    (ismissing(ra) || ismissing(rb)) && return false
    return ra < rb
end
function prefers(x::StrictRank, pool::CandidatePool, a::Symbol, b::Symbol)
    return rank(x, pool, a) < rank(x, pool, b)
end

"""
    indifferent(x, pool, a::Symbol, b::Symbol) -> Bool

True iff both ranked and tied (always false for `StrictRank`).
"""
function indifferent(x::WeakRank, pool::CandidatePool, a::Symbol, b::Symbol)
    ra = rank(x, pool, a); rb = rank(x, pool, b)
    (ismissing(ra) || ismissing(rb)) && return false
    return ra == rb
end
indifferent(::StrictRank, ::CandidatePool, ::Symbol, ::Symbol) = false

"""
    ordered_candidates(x::StrictRank, pool) -> Vector{Symbol}

Return names sorted by strict rank.
"""
function ordered_candidates(x::StrictRank{N}, pool::CandidatePool{N}) where {N}
    perm = sortperm(collect(1:N), by = i -> Int(x.ranks[i]))
    return collect(pool.names[perm])
end

"""
    tie_groups(x::WeakRank, pool) -> Vector{Vector{Symbol}}

Convenience names view of indifference classes.
"""
function tie_groups(x::WeakRank{N}, pool::CandidatePool{N}) where {N}
    wo = to_weakorder(x)
    return weakorder_symbol_groups(wo, pool)
end

"""
    asdict(x::WeakRank, pool) -> Dict{Symbol,Int}
    asdict(x::StrictRank, pool) -> Dict{Symbol,Int}

Emit `{name => rank}` for present ranks only (omits unranked in `WeakRank`).
"""
function asdict(x::WeakRank{N}, pool::CandidatePool{N}) where {N}
    d = Dict{Symbol,Int}()
    @inbounds for i in 1:N
        r = x.ranks[i]
        if !ismissing(r)
            d[pool.names[i]] = Int(r)
        end
    end
    return d
end
function asdict(x::StrictRank{N}, pool::CandidatePool{N}) where {N}
    d = Dict{Symbol,Int}()
    @inbounds for i in 1:N
        d[pool.names[i]] = Int(x.ranks[i])
    end
    return d
end

# ============================ Restriction hooks ===========================

"""
    restrict(pool, subset) -> (new_pool, backmap)

`subset` can be ids or names. Returns `new_pool` and `backmap::SVector{K,CandidateId}` mapping new→old ids.
"""
function restrict(pool::CandidatePool{N}, subset) where {N}
    ids = _resolve_subset(pool, subset)
    return _restrict_pool(pool, ids)
end

"""
    restrict(x::WeakRank, pool, subset)
      -> (WeakRank{K}, new_pool, backmap)

Keep only the subset; present ranks are re-compressed to `1..L` preserving ties.
Unranked remain `missing`.
"""
function restrict(x::WeakRank{N}, pool::CandidatePool{N}, subset) where {N}
    ids = _resolve_subset(pool, subset)
    new_pool, backmap = _restrict_pool(pool, ids)
    K = length(ids)
    sel = Vector{Union{Missing,PreferenceIndex}}(undef, K)
    @inbounds for j in 1:K
        sel[j] = x.ranks[Int(backmap[j])]
    end
    selc = _compress_ranks(sel)
    new_ranks = ntuple(j -> selc[j], Val(K))
    return WeakRank{K}(SVector{K,Union{Missing,PreferenceIndex}}(new_ranks)), new_pool, backmap
end

"""
    restrict(x::StrictRank, pool, subset)
      -> (StrictRank{K}, new_pool, backmap)

Keep subset preserving original relative order; ranks become `1..K`.
"""
function restrict(x::StrictRank{N}, pool::CandidatePool{N}, subset) where {N}
    ids = _resolve_subset(pool, subset)
    new_pool, backmap = _restrict_pool(pool, ids)
    K = length(ids)
    # original permutation order by rank asc
    perm = sortperm(collect(1:N), by = i -> Int(x.ranks[i]))
    keep = Set(Int.(backmap))
    filtered = [i for i in perm if i in keep]
    inv = Dict{Int,PreferenceIndex}(Int(backmap[j]) => PreferenceIndex(j) for j in 1:K)
    ord = [inv[i] for i in filtered]
    ranks = Vector{PreferenceIndex}(undef, K)
    @inbounds for (pos, cid) in enumerate(ord)
        ranks[Int(cid)] = PreferenceIndex(pos)
    end
    return StrictRank{K}(SVector{K,PreferenceIndex}(Tuple(ranks))), new_pool, backmap
end

"""
    restrict(wo::WeakOrder, pool, subset)
      -> (WeakOrder{K}, new_pool, backmap)
"""
function restrict(wo::WeakOrder{N}, pool::CandidatePool{N}, subset) where {N}
    ids = _resolve_subset(pool, subset)
    new_pool, backmap = _restrict_pool(pool, ids)
    K = length(ids)
    inv = Dict{Int,PreferenceIndex}(Int(backmap[j]) => PreferenceIndex(j) for j in 1:K)
    keep = Set(Int.(ids))
    new_levels = Vector{Vector{CandidateId}}()
    for lvl in wo.levels
        lvl2 = CandidateId[]
        for cid in lvl
            if Int(cid) in keep
                push!(lvl2, inv[Int(cid)])
            end
        end
        if !isempty(lvl2)
            push!(new_levels, lvl2)
        end
    end
    return WeakOrder{K}(new_levels), new_pool, backmap
end

"""
    restrict(pp::Pairwise, pool, subset)
      -> (Pairwise{K}, new_pool, backmap)
"""
function restrict(pp::Pairwise{N,T}, pool::CandidatePool{N}, subset) where {N,T<:Integer}
    ids = _resolve_subset(pool, subset)
    new_pool, backmap = _restrict_pool(pool, ids)
    K = length(ids)
    data = ntuple(i -> ntuple(j -> pp.matrix[Int(backmap[i]), Int(backmap[j])], Val(K)), Val(K))
    return Pairwise{K,T}(SMatrix{K,K,T}(data)), new_pool, backmap
end

# ---- subset helpers ----

"""
    _resolve_subset(pool, subset) -> Vector{CandidateId}

Accepts ids or names; validates non-empty, no duplicates, id bounds.
"""
function _resolve_subset(pool::CandidatePool{N}, subset) where {N}
    ids = CandidateId[]
    if eltype(subset) <: Symbol
        for s in subset
            push!(ids, pool[s])
        end
    else
        for x in subset
            cid = CandidateId(x)
            (cid < 1 || cid > CandidateId(N)) && throw(ArgumentError("Candidate id $(x) out of bounds 1..$N"))
            push!(ids, cid)
        end
    end
    isempty(ids) && throw(ArgumentError("Restriction subset cannot be empty"))
    length(Set(ids)) == length(ids) || throw(ArgumentError("Restriction subset has duplicates"))
    return ids
end

"""
    _restrict_pool(pool, ids) -> (new_pool, backmap)

Make a new `CandidatePool` for the subset and a static `backmap` (new→old).
"""
function _restrict_pool(pool::CandidatePool{N}, ids::Vector{CandidateId}) where {N}
    K = length(ids)
    new_names = [pool.names[Int(i)] for i in ids]
    new_pool = CandidatePool(new_names)
    backmap = SVector{K,CandidateId}(Tuple(ids))
    return new_pool, backmap
end

"""
    _compress_ranks(selected::Vector{Union{Missing,PreferenceIndex}}) -> Vector{Union{Missing,PreferenceIndex}}

Map present ranks to `1..L` preserving ties; keep `missing` as `missing`.
"""
function _compress_ranks(selected::Vector{Union{Missing,PreferenceIndex}})
    present = unique(filter(!ismissing, selected))
    sort!(present)
    m = Dict{PreferenceIndex,PreferenceIndex}(r => PreferenceIndex(i) for (i,r) in enumerate(present))
    out = Vector{Union{Missing,PreferenceIndex}}(undef, length(selected))
    @inbounds for i in eachindex(selected)
        r = selected[i]
        out[i] = ismissing(r) ? missing : m[PreferenceIndex(r)]
    end
    return out
end

# ============================ CSV Readers ================================

"""
    read_rank_columns_csv(path; candidates=nothing, missingstrings=["", "NA", "NaN"])
      -> (ballots::Vector{WeakRank}, pool::CandidatePool)

CSV where each candidate is a column and cells contain a rank (or blank).
- If `candidates` is given, validates and uses that order; otherwise uses file column order.
- Returns `WeakRank` ballots; strict users can call `to_strict` as needed.
Requires `CSV.jl` at runtime.
"""
function read_rank_columns_csv(path::AbstractString;
                               candidates::Union{Nothing,Vector{Symbol}}=nothing,
                               missingstrings = ["", "NA", "NaN"])
    CSV = _require_CSV()
    tbl = CSV.File(path; missingstring=missingstrings, silencewarnings=true)
    cols_all = Symbol.(propertynames(tbl))
    cand_syms = candidates === nothing ? cols_all : begin
        missing_cols = setdiff(candidates, cols_all)
        !isempty(missing_cols) && throw(ArgumentError("Missing candidate columns: $(collect(missing_cols))"))
        candidates
    end
    pool = CandidatePool(cand_syms)
    ballots = WeakRank[]
    for row in tbl
        d = Dict{Symbol,Int}()
        for c in cand_syms
            v = row[c]
            if v !== missing
                d[c] = Int(v)
            end
        end
        push!(ballots, WeakRank(pool, d))
    end
    return ballots, pool
end

"""
    read_candidate_columns_csv(path; cols, pool=nothing, missingstrings=["", "NA", "NaN"])
      -> (ballots::Vector{WeakRank}, pool::CandidatePool)

CSV where **columns encode ranked positions** (leftmost = top choice). Cells are names.
- `cols` must exist in the file (validated).
- If `pool` omitted, it is inferred from the union of names (sorted for determinism).
- Ties are not representable here; blanks imply truncation.
Returns `WeakRank` for uniformity.
"""
function read_candidate_columns_csv(path::AbstractString;
                                    cols::Vector{Symbol},
                                    pool::Union{Nothing,CandidatePool}=nothing,
                                    missingstrings = ["", "NA", "NaN"])
    CSV = _require_CSV()
    tbl = CSV.File(path; missingstring=missingstrings, silencewarnings=true)

    file_cols = Symbol.(propertynames(tbl))
    missing_cols = setdiff(cols, file_cols)
    !isempty(missing_cols) && throw(ArgumentError("Missing ordered columns: $(collect(missing_cols))"))

    # infer pool if needed
    local cp
    if pool === nothing
        seen = Set{Symbol}()
        for row in tbl
            for c in cols
                v = row[c]
                if v !== missing
                    s = v isa Symbol ? v : Symbol(String(v))
                    push!(seen, s)
                end
            end
        end
        cp = CandidatePool(sort!(collect(seen)))
        tbl = CSV.File(path; missingstring=missingstrings, silencewarnings=true) # reopen
    else
        cp = pool
    end

    ballots = WeakRank[]
    for row in tbl
        # build ranks incrementally from left to right
        ordered_syms = Symbol[]
        for c in cols
            v = row[c]
            if v !== missing
                s = v isa Symbol ? v : Symbol(String(v))
                push!(ordered_syms, s)
            end
        end
        if isempty(ordered_syms)
            push!(ballots, WeakRank(cp, Dict{Symbol,Int}()))  # all missing
        else
            # convert ordered names to a WeakRank with ranks 1..k for the listed names
            N = length(cp)
            vr = Vector{Union{Missing,PreferenceIndex}}(undef, N)
            fill!(vr, missing)
            for (pos, nm) in enumerate(ordered_syms)
                vr[Int(cp[nm])] = PreferenceIndex(pos)
            end
            push!(ballots, WeakRank{N}(SVector{N,Union{Missing,PreferenceIndex}}(ntuple(i->vr[i], Val(N)))))
        end
    end
    return ballots, cp
end

# ============================ CSV dependency =============================

"""
    _require_CSV() -> Module

Lazy-import CSV.jl or throw a helpful error.
"""
@inline function _require_CSV()
    try
        @eval import CSV
        return CSV
    catch
        throw(ArgumentError("read_*_csv requires CSV.jl. Please: using Pkg; Pkg.add(\"CSV\"); then retry."))
    end
end

# ============================ Exports (commented) =========================
# export CandidatePool, WeakRank, StrictRank, Pairwise, WeakOrder,
#        candidates, to_cmap, candidateat,
#        to_weak, to_strict, to_perm, to_pairwise, to_weakorder,
#        rank, prefers, indifferent, ordered_candidates, tie_groups, asdict,
#        restrict,
#        read_rank_columns_csv, read_candidate_columns_csv
