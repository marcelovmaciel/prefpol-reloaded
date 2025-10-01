# src/preferences_types.jl
using StaticArrays: SVector, SMatrix

const PreferenceIndex = UInt16
const CandidateId = PreferenceIndex  # explicit alias for clarity when referring to ids

"""
    CandidatePool{N}

Immutable mapping between candidate *names* (symbols) and integer *ids* `1..N`.

Design (pragmatic hybrid):
- `names::SVector{N,Symbol}` stores the canonical, ordered list of candidates.
- `index::Dict{Symbol,CandidateId}` maps name → id.

Why this design?
- Keeps `N` static for stack-allocated small arrays and good inference.
- Avoids proliferating a distinct `CandidatePool{Names,N}` type per name set, reducing compilation and invalidations in exploratory workflows.

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

"""
    CandidatePool(names::AbstractVector{Symbol})
    CandidatePool(names::AbstractVector{<:AbstractString})

Create a `CandidatePool` from unique candidate names. Throws if empty or duplicated.
"""
function CandidatePool(names::AbstractVector{Symbol})
    cand_tuple = Tuple(names)
    N = length(cand_tuple)
    N == 0 && throw(ArgumentError("Candidate pool cannot be empty"))
    if length(Set(cand_tuple)) != N
        throw(ArgumentError("Candidates must be unique"))
    end
    name_vec = SVector{N,Symbol}(cand_tuple)
    # deterministic 1..N assignment in the canonical order
    idx = Dict{Symbol,CandidateId}(s => CandidateId(i) for (i, s) in enumerate(name_vec))
    CandidatePool{N}(name_vec, idx)
end

CandidatePool(names::AbstractVector{<:AbstractString}) =
    CandidatePool(Symbol.(names))

Base.length(pool::CandidatePool{N}) where {N} = N

function Base.iterate(pool::CandidatePool{N}, state::Int=1) where {N}
    state > N && return nothing
    return pool.names[state], state + 1
end

"""
    pool[sym::Symbol] -> CandidateId

Name → id lookup.
"""
Base.getindex(pool::CandidatePool, cand::Symbol) = pool.index[cand]

"""
    pool[i::Integer] -> Symbol

Id → name lookup.
"""
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

# ------------------------- Preferences core types -------------------------

abstract type PreferenceStyle end
"""
Tag for linear/strict orders (no ties)."""
struct Strict <: PreferenceStyle end
"""
Tag for weak orders (ties allowed)."""
struct Weak   <: PreferenceStyle end

abstract type AbstractPreference{Style<:PreferenceStyle,N} end

"""
    RankPreference{Style,N,R<:Unsigned} <: AbstractPreference{Style,N}

Ballot encoded as *ranks* per candidate id `1..N`. Lower rank = better.

- `ranks::SVector{N,Union{Missing,R}}`:
  - `missing` = unranked (truncation).
  - Present ranks must be integers `≥ 1`. Ties allowed iff `Style === Weak`.

Construct via helpers `RankPreference(pool, ...)` below.
"""
struct RankPreference{Style<:PreferenceStyle,N,R<:Unsigned} <: AbstractPreference{Style,N}
    ranks::SVector{N,Union{Missing,R}}  # missing = unranked
end

"""
    OrderPreference{Style,N,R<:Unsigned} <: AbstractPreference{Style,N}

Ballot encoded as a *permutation* (strict order only).
"""
struct OrderPreference{Style<:PreferenceStyle,N,R<:Unsigned} <: AbstractPreference{Style,N}
    order::SVector{N,R}
end

"""
    PairwisePreference{Style,N,T<:Integer} <: AbstractPreference{Style,N}

Pairwise comparison *sign* matrix at the ballot level:
- entry `+1` if `i ≻ j`, `-1` if `j ≻ i`, `0` if tie/no-comparison (e.g., due to truncation policy).
"""
struct PairwisePreference{Style<:PreferenceStyle,N,T<:Integer} <: AbstractPreference{Style,N}
    matrix::SMatrix{N,N,T}
end

# ==================== WeakOrder (indifference classes) ====================

"""
    WeakOrder{N}

First-class weak order as levels of tied candidates.

- `levels::Vector{Vector{CandidateId}}` with `levels[1]` the top indifference class, etc.
- Every candidate `1..N` must appear *exactly once*.
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
        if length(seen) != N
            throw(ArgumentError("WeakOrder must cover all $N candidates exactly once"))
        end
        new{N}(levels)
    end
end

Base.length(::WeakOrder{N}) where {N} = N
Base.iterate(wo::WeakOrder{N}, state::Int=1) where {N} = state > N ? nothing : (wo.levels[state], state+1)
function Base.show(io::IO, wo::WeakOrder{N}) where {N}
    print(io, "WeakOrder{$N}(", wo.levels, ")")
end

"""
    weakorder_symbol_groups(wo, pool) -> Vector{Vector{Symbol}}

Render levels of a `WeakOrder` as candidate names.
"""
function weakorder_symbol_groups(wo::WeakOrder{N}, pool::CandidatePool{N}) where {N}
    [map(i -> pool.names[Int(i)], lvl) for lvl in wo.levels]
end

# ------------------------- Internal helpers -------------------------

@inline _nonmissing(xs) = (x for x in xs if !ismissing(x))
@inline function _min_pos_ignore_missing(xs)
    m = typemax(Int)
    found = false
    for x in xs
        if !ismissing(x)
            found = true
            m = min(m, Int(x))
        end
    end
    return found ? m : nothing
end

"""
    _style_type(style) -> Type{Strict}|Type{Weak}

Resolve user style input (`:strict`, `:weak`, `Strict`, `Weak`) to a concrete type.
"""
@inline function _style_type(style)
    style === :strict && return Strict
    style === :weak   && return Weak
    style === Strict  && return Strict
    style === Weak    && return Weak
    style <: PreferenceStyle && return style
    throw(ArgumentError("Unknown preference style $(style)"))
end

"""
    _to_prefindex_or_missing(x) -> Union{Missing,PreferenceIndex}

Safely convert a user rank value to `PreferenceIndex`, rejecting non-positive integers.
`missing` is passed through unchanged.
"""
@inline function _to_prefindex_or_missing(x)
    ismissing(x) && return missing
    xi = Int(x)
    xi < 1 && throw(ArgumentError("Ranks must be ≥ 1, got $xi"))
    return PreferenceIndex(xi)
end

"""
    _maybe_dict_rank(dict, cand) -> Union{Missing,PreferenceIndex}

Read `cand` from a dictionary of ranks; missing key => `missing`.
Values are validated via `_to_prefindex_or_missing`.
"""
@inline function _maybe_dict_rank(dict::AbstractDict{Symbol,<:Integer}, cand::Symbol)
    if haskey(dict, cand)
        return _to_prefindex_or_missing(dict[cand])
    else
        return missing
    end
end

"""
    _ensure_valid(::Type{Strict|Weak}, ranks_tuple)

Validate a ranks tuple (with possible `missing`) according to the style.
- Strict: all present ranks unique; all present ≥ 1.
- Weak: all present ≥ 1.
"""
@inline function _ensure_valid(::Type{Strict}, ranks::NTuple{N,Union{Missing,PreferenceIndex}}) where {N}
    vals = collect(_nonmissing(ranks))
    if length(Set(vals)) != length(vals)
        throw(ArgumentError("Strict preferences require unique assigned ranks"))
    end
    m = _min_pos_ignore_missing(vals)
    if m !== nothing && m < 1
        throw(ArgumentError("Ranks must be ≥ 1"))
    end
    return nothing
end

@inline function _ensure_valid(::Type{Weak}, ranks::NTuple{N,Union{Missing,PreferenceIndex}}) where {N}
    m = _min_pos_ignore_missing(ranks)
    if m !== nothing && m < 1
        throw(ArgumentError("Ranks must be ≥ 1"))
    end
    return nothing
end

"""
    _detect_style(ranks_tuple) -> Type{Strict|Weak}

Infer style from present ranks: unique ⇒ `Strict`; duplicates ⇒ `Weak`.
"""
@inline function _detect_style(ranks::NTuple{N,Union{Missing,PreferenceIndex}}) where {N}
    vals = collect(_nonmissing(ranks))
    return (length(Set(vals)) == length(vals)) ? Strict : Weak
end

"""
    _collect_ranks(pool, dict) -> NTuple{N,Union{Missing,PreferenceIndex}}

Collect ranks for all candidates in `pool` from a dict `{name => rank}`.
"""
@inline function _collect_ranks(pool::CandidatePool{N},
                                dict::AbstractDict{Symbol,<:Integer}) where {N}
    ntuple(i -> _maybe_dict_rank(dict, pool.names[i]), Val(N))
end

# ------------------------- RankPreference constructors -------------------------

"""
    RankPreference{Style}(pool, ranks_tuple)

Construct from a tuple of validated `Union{Missing,PreferenceIndex}`.
"""
RankPreference{Style}(pool::CandidatePool{N},
                      ranks::NTuple{N,Union{Missing,PreferenceIndex}}) where {Style<:PreferenceStyle,N} =
    ( _ensure_valid(Style, ranks);
      RankPreference{Style,N,PreferenceIndex}(SVector{N,Union{Missing,PreferenceIndex}}(ranks)) )

"""
    RankPreference{Style}(pool, ranks_tuple::NTuple{N,<:Integer})

Construct from integer ranks; values are validated (`≥ 1`) then converted.
"""
RankPreference{Style}(pool::CandidatePool{N},
                      ranks::NTuple{N,<:Integer}) where {Style<:PreferenceStyle,N} =
    RankPreference{Style}(pool, ntuple(i -> _to_prefindex_or_missing(ranks[i]), Val(N)))

"""
    RankPreference{Style}(pool, ranks::SVector{N,Union{Missing,<:Integer}})

Construct from an SVector with integers or `missing`.
"""
RankPreference{Style}(pool::CandidatePool{N},
                      ranks::SVector{N,Union{Missing,<:Integer}}) where {Style<:PreferenceStyle,N} =
    RankPreference{Style}(pool, ntuple(i -> _to_prefindex_or_missing(ranks[i]), Val(N)))

"""
    RankPreference{Style}(pool, ranks::SVector{N,<:Integer})  # no missing

Construct from an SVector of integers.
"""
RankPreference{Style}(pool::CandidatePool{N},
                      ranks::SVector{N,<:Integer}) where {Style<:PreferenceStyle,N} =
    RankPreference{Style}(pool, Tuple(ranks))

"""
    RankPreference(pool, dict; style=:auto)

Construct from `{name => rank}` dict. Missing keys ⇒ `missing` ranks.
"""
function RankPreference(pool::CandidatePool{N},
                        dict::AbstractDict{Symbol,<:Integer};
                        style::Union{Symbol,Type}= :auto) where {N}
    ranks = _collect_ranks(pool, dict)
    Style = style === :auto ? _detect_style(ranks) : _style_type(style)
    return RankPreference{Style}(pool, ranks)
end

"""
    RankPreference(pool, ranks::AbstractVector{<:Integer}; style=:auto)

Construct from a dense vector of integer ranks (no truncation).
"""
function RankPreference(pool::CandidatePool{N},
                        ranks::AbstractVector{<:Integer};
                        style::Union{Symbol,Type}= :auto) where {N}
    length(ranks) == N || throw(ArgumentError("Expected $(N) ranks, got $(length(ranks))"))
    ranks_tuple = ntuple(i -> _to_prefindex_or_missing(ranks[i]), Val(N))
    Style = style === :auto ? _detect_style(ranks_tuple) : _style_type(style)
    return RankPreference{Style}(pool, ranks_tuple)
end

"""
    RankPreference(pool, ranks::AbstractVector{T}; style=:auto)

Construct from a vector that may contain `missing`. Non-missing validated to be ≥ 1.
"""
function RankPreference(pool::CandidatePool{N},
                        ranks::AbstractVector{T};
                        style::Union{Symbol,Type}= :auto) where {N,T}
    length(ranks) == N || throw(ArgumentError("Expected $(N) ranks, got $(length(ranks))"))
    ranks_tuple = ntuple(i -> _to_prefindex_or_missing(ranks[i]), Val(N))
    Style = style === :auto ? _detect_style(ranks_tuple) : _style_type(style)
    return RankPreference{Style}(pool, ranks_tuple)
end

"""
    OrderPreference(pool, order_names)

Strict order constructed from a full permutation of candidate names.
"""
function OrderPreference(pool::CandidatePool{N},
                         order::AbstractVector{Symbol}) where {N}
    length(order) == N || throw(ArgumentError("Expected $(N) candidates, got $(length(order))"))
    idxs = ntuple(i -> PreferenceIndex(pool[order[i]]), Val(N))
    length(Set(idxs)) == N || throw(ArgumentError("Order must mention each candidate once"))
    return OrderPreference{Strict,N,PreferenceIndex}(SVector{N,PreferenceIndex}(idxs))
end

"""
    OrderPreference(pref::RankPreference{Strict}, pool)

Derive a strict order from strict ranks (ties absent). Ranked first by `(rank)`, then index.
Unranked entries (shouldn't occur for strict) are placed last defensively.
"""
function OrderPreference(pref::RankPreference{Strict,N,R},
                         pool::CandidatePool{N}) where {N,R}
    perm = sortperm(collect(1:N),
                    by = i -> begin
                        r = pref.ranks[i]
                        ismissing(r) ? (1, typemax(Int), i) : (0, Int(r), i)
                    end)
    idxs = ntuple(i -> PreferenceIndex(perm[i]), Val(N))
    return OrderPreference{Strict,N,PreferenceIndex}(SVector{N,PreferenceIndex}(idxs))
end

# ================= Pairwise extraction with extension policy =================

"""
    ext_bottom(ra, rb, i, j, ranks, pool) -> Int

Default extension: ranked ≻ unranked. Both unranked ⇒ tie (0).
"""
@inline function ext_bottom(ra, rb, i, j, ranks, pool)
    if ismissing(ra) && ismissing(rb)
        return 0
    elseif ismissing(ra)
        return -1
    elseif ismissing(rb)
        return 1
    else
        d = Int(rb) - Int(ra) # lower rank = better
        return d > 0 ? 1 : d < 0 ? -1 : 0
    end
end

"""
    ext_none(ra, rb, i, j, ranks, pool) -> Int

Strict-only comparisons: any `missing` ⇒ no-comparison (0).
"""
@inline function ext_none(ra, rb, i, j, ranks, pool)
    if ismissing(ra) || ismissing(rb)
        return 0
    else
        d = Int(rb) - Int(ra)
        return d > 0 ? 1 : d < 0 ? -1 : 0
    end
end

"""
    _resolve_extension(extension) -> Function

Accepts `:bottom`, `:none`, or a custom `extension(ra,rb,i,j,ranks,pool)::Int`.
"""
@inline function _resolve_extension(extension)
    if extension === :bottom || extension === nothing
        return ext_bottom
    elseif extension === :none
        return ext_none
    elseif extension isa Function
        return extension
    else
        throw(ArgumentError("Unknown extension method $(extension). Use :bottom, :none, or a function."))
    end
end

"""
    PairwisePreference(pref, pool; extension=:bottom) -> PairwisePreference

Build pairwise sign matrix for a single ballot using the chosen extension/imputation policy.
"""
function PairwisePreference(pref::RankPreference{Style,N,R},
                            pool::CandidatePool{N};
                            extension = :bottom) where {Style<:PreferenceStyle,N,R}
    ext = _resolve_extension(extension)
    data = ntuple(i -> ntuple(j -> _pairwise_entry(pref.ranks, i, j, ext, pool), Val(N)), Val(N))
    return PairwisePreference{Style,N,Int8}(SMatrix{N,N,Int8}(data))
end

@inline function _pairwise_entry(ranks::SVector{N,Union{Missing,R}},
                                 i::Int, j::Int, ext, pool) where {N,R}
    i == j && return Int8(0)
    ai = ranks[i]; bj = ranks[j]
    return Int8(ext(ai, bj, i, j, ranks, pool))
end

# ------------------------- Predicates & utilities -------------------------

"""
    rank(pref, pool, name::Symbol) -> Union{Missing,PreferenceIndex}

Lookup a candidate’s rank in a ballot (may be `missing`).
"""
function rank(pref::RankPreference{Style,N,R},
              pool::CandidatePool{N},
              cand::Symbol) where {Style<:PreferenceStyle,N,R}
    return pref.ranks[pool[cand]]
end

"""
    prefers(pref, pool, a::Symbol, b::Symbol) -> Bool

True iff both ranked and `a` has a strictly smaller rank than `b`.
"""
function prefers(pref::RankPreference, pool::CandidatePool, a::Symbol, b::Symbol)
    ra = rank(pref, pool, a); rb = rank(pref, pool, b)
    (ismissing(ra) || ismissing(rb)) && return false
    return ra < rb
end

"""
    indifferent(pref, pool, a::Symbol, b::Symbol) -> Bool

True iff both ranked and tied.
"""
function indifferent(pref::RankPreference, pool::CandidatePool, a::Symbol, b::Symbol)
    ra = rank(pref, pool, a); rb = rank(pref, pool, b)
    (ismissing(ra) || ismissing(rb)) && return false
    return ra == rb
end

"""
    ordered_candidates(pref::RankPreference{Strict}, pool) -> Vector{Symbol}

Return names sorted by strict rank; unranked (defensive) at the end.
"""
function ordered_candidates(pref::RankPreference{Strict,N,R},
                            pool::CandidatePool{N}) where {N,R}
    perm = sortperm(collect(1:N),
                    by = i -> begin
                        r = pref.ranks[i]
                        ismissing(r) ? (1, typemax(Int), i) : (0, Int(r), i)
                    end)
    return collect(pool.names[perm])
end

# ------------------------- WeakOrder construction & conversions -------------------------

"""
    weakorder(pref::RankPreference{Weak}) -> WeakOrder

Convert weak ranks (with truncation) into an explicit levels representation.
Unranked candidates are placed in the last level so coverage is total.
"""
function weakorder(pref::RankPreference{Weak,N,R}) where {N,R}
    groups = Dict{PreferenceIndex,Vector{CandidateId}}()
    unranked = CandidateId[]
    @inbounds for i in 1:N
        r = pref.ranks[i]
        if ismissing(r)
            push!(unranked, CandidateId(i))
        else
            push!(get!(groups, PreferenceIndex(r), CandidateId[]), CandidateId(i))
        end
    end
    ranked_levels = [groups[k] for k in sort!(collect(keys(groups)))]
    levels = isempty(unranked) ? ranked_levels : vcat(ranked_levels, [unranked])
    return WeakOrder{N}(levels)
end

"""
    indifference_list(pref::RankPreference{Weak}) -> Vector{Vector{CandidateId}}

Lossless ID-based indifference classes (levels). Use `weakorder_symbol_groups` to view names.
"""
function indifference_list(pref::RankPreference{Weak,N,R}) where {N,R}
    return weakorder(pref).levels
end

"""
    tie_groups(pref::RankPreference{Weak}, pool) -> Vector{Vector{Symbol}}

Backward-compatible symbol rendering for ties.
"""
function tie_groups(pref::RankPreference{Weak,N,R},
                    pool::CandidatePool{N}) where {N,R}
    wo = weakorder(pref)
    return weakorder_symbol_groups(wo, pool)
end

"""
    ranks_from_weakorder(wo) -> NTuple{N,PreferenceIndex}

Assign equal rank to all ids within the same level, level order = rank order.
"""
function ranks_from_weakorder(wo::WeakOrder{N}) where {N}
    ranks = Vector{PreferenceIndex}(undef, N)
    @inbounds for (lvl, ids) in enumerate(wo.levels)
        for cid in ids
            ranks[Int(cid)] = PreferenceIndex(lvl)
        end
    end
    return ntuple(i -> ranks[i], Val(N))
end

"""
    linearize(wo::WeakOrder) -> NTuple{N,PreferenceIndex}

Produce strict ranks by ordering levels then IDs within each level.
"""
function linearize(wo::WeakOrder{N}) where {N}
    ranks = Vector{PreferenceIndex}(undef, N)
    pos = 1
    @inbounds for ids in wo.levels
        for cid in sort(ids)
            ranks[Int(cid)] = PreferenceIndex(pos)
            pos += 1
        end
    end
    return ntuple(i -> ranks[i], Val(N))
end

"""
    linearize(pref::RankPreference{Weak}, pool) -> RankPreference{Strict}

Tie-break by `(rank, id)` then assign 1..N.
"""
function linearize(pref::RankPreference{Weak,N,R},
                   pool::CandidatePool{N}) where {N,R}
    order = collect(1:N)
    sort!(order, by = i -> (ismissing(pref.ranks[i]) ? (true,  typemax(Int), i)
                                                     : (false, Int(pref.ranks[i]), i)))
    ranks = Vector{PreferenceIndex}(undef, N)
    @inbounds for (pos, idx) in enumerate(order)
        ranks[idx] = PreferenceIndex(pos)
    end
    return RankPreference{Strict}(pool, ntuple(i -> ranks[i], Val(N)))
end

# ------------------------- Conversions (dispatch-based) -------------------------

@inline function _order_to_rank_tuple(order::OrderPreference{Strict,N,R}) where {N,R}
    inv = Vector{PreferenceIndex}(undef, N)
    @inbounds for (pos, idx) in enumerate(order.order)
        inv[Int(idx)] = PreferenceIndex(pos)
    end
    return ntuple(i -> inv[i], Val(N))
end

# -- WeakOrder builder from any ranks (with possible missing) --
@inline function _weakorder_from_any_ranks(ranks::SVector{N,Union{Missing,R}}) where {N,R}
    groups = Dict{Int,Vector{CandidateId}}()
    unranked = CandidateId[]
    @inbounds for i in 1:N
        r = ranks[i]
        if ismissing(r)
            push!(unranked, CandidateId(i))
        else
            push!(get!(groups, Int(r), CandidateId[]), CandidateId(i))
        end
    end
    ranked_levels = [groups[k] for k in sort!(collect(keys(groups)))]
    levels = isempty(unranked) ? ranked_levels : vcat(ranked_levels, [unranked])
    return WeakOrder{N}(levels)
end

# ======================== to_rank ========================
"""
    to_rank(::Type{Style}, pool, x) -> RankPreference{Style}
    to_rank(pool, x)                 -> RankPreference{Strict|Weak}  (auto)

Convert inputs to rank-encoded ballots. `to_rank(Style, ...)` forces Style; `to_rank(pool, ...)` infers.
"""
# Style-forcing entry points
to_rank(::Type{S}, pool::CandidatePool{N}, x) where {S<:PreferenceStyle,N} = _to_rank_style(S, pool, x)
# Auto-style entry points
to_rank(pool::CandidatePool{N}, x) where {N} = _to_rank_auto(pool, x)

# --- Style-forcing methods ---
_to_rank_style(::Type{S}, pool::CandidatePool{N}, x::RankPreference{S,N,R}) where {S<:PreferenceStyle,N,R} = x
_to_rank_style(::Type{Strict}, pool::CandidatePool{N}, x::RankPreference{Weak,N,R}) where {N,R} = linearize(x, pool)
_to_rank_style(::Type{Weak},   pool::CandidatePool{N}, x::RankPreference{Strict,N,R}) where {N,R} =
    RankPreference{Weak}(pool, ntuple(i -> (r = x.ranks[i]; ismissing(r) ? missing : PreferenceIndex(r)), Val(N)))

_to_rank_style(::Type{S}, pool::CandidatePool{N}, d::AbstractDict{Symbol,<:Integer}) where {S<:PreferenceStyle,N} =
    RankPreference(pool, d; style=S)
_to_rank_style(::Type{S}, pool::CandidatePool{N}, v::AbstractVector) where {S<:PreferenceStyle,N} =
    RankPreference(pool, v; style=S)

_to_rank_style(::Type{Weak},   pool::CandidatePool{N}, wo::WeakOrder{N}) where {N} =
    RankPreference{Weak}(pool, ranks_from_weakorder(wo))
_to_rank_style(::Type{Strict}, pool::CandidatePool{N}, wo::WeakOrder{N}) where {N} =
    RankPreference{Strict}(pool, linearize(wo))

_to_rank_style(::Type{S}, pool::CandidatePool{N}, ord::OrderPreference{Strict,N,R}) where {S<:PreferenceStyle,N,R} =
    RankPreference{S}(pool, _order_to_rank_tuple(ord))

# --- Auto-style methods ---
_to_rank_auto(pool::CandidatePool{N}, x::RankPreference{Style,N,R}) where {Style<:PreferenceStyle,N,R} = x
_to_rank_auto(pool::CandidatePool{N}, d::AbstractDict{Symbol,<:Integer}) where {N} =
    RankPreference(pool, d; style=:auto)
_to_rank_auto(pool::CandidatePool{N}, v::AbstractVector) where {N} =
    RankPreference(pool, v; style=:auto)
_to_rank_auto(pool::CandidatePool{N}, wo::WeakOrder{N}) where {N} =
    RankPreference{Weak}(pool, ranks_from_weakorder(wo))
_to_rank_auto(pool::CandidatePool{N}, ord::OrderPreference{Strict,N,R}) where {N,R} =
    RankPreference{Strict}(pool, _order_to_rank_tuple(ord))

# ======================== to_order ========================
"""
    to_order(pool, x) -> OrderPreference{Strict}

Produce a strict permutation. Weak/tied inputs are linearized.
"""
to_order(pool::CandidatePool{N}, x::OrderPreference{Strict,N,R}) where {N,R} = x
to_order(pool::CandidatePool{N}, x::RankPreference{Strict,N,R}) where {N,R} = OrderPreference(x, pool)
to_order(pool::CandidatePool{N}, x::RankPreference{Weak,N,R})   where {N,R} = OrderPreference(linearize(x, pool), pool)
to_order(pool::CandidatePool{N}, wo::WeakOrder{N})              where {N}   = OrderPreference(RankPreference{Strict}(pool, linearize(wo)), pool)
to_order(pool::CandidatePool{N}, d::AbstractDict{Symbol,<:Integer}) where {N} = OrderPreference(to_rank(Strict, pool, d), pool)
to_order(pool::CandidatePool{N}, v::AbstractVector)                where {N} = OrderPreference(to_rank(Strict, pool, v), pool)

# ======================== to_weakorder ========================
"""
    to_weakorder(x) -> WeakOrder

Convert ranks or strict order to an explicit weak-order (levels). Unranked become last level.
"""
to_weakorder(wo::WeakOrder{N})                    where {N}   = wo
to_weakorder(rp::RankPreference{Weak,N,R})        where {N,R} = weakorder(rp)
to_weakorder(rp::RankPreference{Strict,N,R})      where {N,R} = _weakorder_from_any_ranks(rp.ranks)
to_weakorder(ord::OrderPreference{Strict,N,R})    where {N,R} = WeakOrder{N}([[CandidateId(ord.order[k])] for k in 1:N])

# ======================== to_pairwise ========================
"""
    to_pairwise(pool, x; extension=:bottom) -> PairwisePreference{Style}
    to_pairwise(::Type{Style}, pool, x; extension=:bottom) -> PairwisePreference{Style}

Build pairwise sign matrices. Style comes from `x` if rank-like; otherwise from `to_rank`’s auto/forced style.
"""
to_pairwise(pool::CandidatePool{N}, rp::RankPreference{Style,N,R}; extension=:bottom) where {Style<:PreferenceStyle,N,R} =
    PairwisePreference(rp, pool; extension=extension)

to_pairwise(pool::CandidatePool{N}, x; extension=:bottom) where {N} =
    to_pairwise(pool, to_rank(pool, x); extension=extension)

to_pairwise(::Type{S}, pool::CandidatePool{N}, x; extension=:bottom) where {S<:PreferenceStyle,N} =
    to_pairwise(pool, to_rank(S, pool, x); extension=extension)

# ------------------------- Direct convenience constructors -------------------------
RankPreference{Weak}(pool::CandidatePool{N}, wo::WeakOrder{N}) where {N} =
    RankPreference{Weak}(pool, ranks_from_weakorder(wo))
RankPreference{Strict}(pool::CandidatePool{N}, wo::WeakOrder{N}) where {N} =
    RankPreference{Strict}(pool, linearize(wo))
RankPreference{Style}(pool::CandidatePool{N}, ord::OrderPreference{Strict,N,R}) where {Style<:PreferenceStyle,N,R} =
    RankPreference{Style}(pool, _order_to_rank_tuple(ord))

# ------------------------- Restriction hooks -------------------------

"""
    restrict(pool, curr_cands) -> (new_pool, backmap)

Restrict a `CandidatePool` to `curr_cands` (ids or names), preserving order.
Returns a new pool and a `backmap::SVector{K,CandidateId}` (new id → old id).
"""
function restrict(pool::CandidatePool{N}, curr_cands) where {N}
    ids = _resolve_subset(pool, curr_cands)
    _restrict_pool(pool, ids)
end

"""
    restrict(pref::RankPreference, pool, curr_cands)
      -> (new_pref::RankPreference{Style,K}, new_pool, backmap)

Restrict ranks to a candidate subset. Present ranks are re-compressed to contiguous
`1..L` preserving ties; `missing` remains `missing`.
"""
function restrict(pref::RankPreference{Style,N,R},
                  pool::CandidatePool{N},
                  curr_cands) where {Style<:PreferenceStyle,N,R}
    ids = _resolve_subset(pool, curr_cands)
    new_pool, backmap = _restrict_pool(pool, ids)
    K = length(ids)
    sel = Vector{Union{Missing,PreferenceIndex}}(undef, K)
    @inbounds for j in 1:K
        sel[j] = pref.ranks[Int(backmap[j])]
    end
    selc = _compress_ranks(sel)
    new_ranks = ntuple(j -> selc[j], Val(K))
    new_pref = RankPreference{Style}(new_pool, new_ranks)
    return new_pref, new_pool, backmap
end

"""
    restrict(order::OrderPreference{Strict}, pool, curr_cands)
      -> (new_order::OrderPreference{Strict,K}, new_pool, backmap)

Restrict a strict order to a subset, preserving original relative order.
"""
function restrict(order::OrderPreference{Strict,N,R},
                  pool::CandidatePool{N},
                  curr_cands) where {N,R}
    ids = _resolve_subset(pool, curr_cands)
    new_pool, backmap = _restrict_pool(pool, ids)
    K = length(ids)
    orig = collect(Int.(order.order))
    keep = Set(Int.(ids))
    filtered = Int[]
    @inbounds for idx in orig
        if idx in keep
            push!(filtered, idx)
        end
    end
    inv = Dict{Int,PreferenceIndex}(Int(backmap[j]) => PreferenceIndex(j) for j in 1:K)
    new_order_vec = [inv[i] for i in filtered]
    new_order = OrderPreference{Strict,K,PreferenceIndex}(SVector{K,PreferenceIndex}(Tuple(new_order_vec)))
    return new_order, new_pool, backmap
end

"""
    restrict(wo::WeakOrder, pool, curr_cands)
      -> (new_wo::WeakOrder{K}, new_pool, backmap)

Filter each level to the subset, drop empties, and reindex ids to `1..K`.
"""
function restrict(wo::WeakOrder{N},
                  pool::CandidatePool{N},
                  curr_cands) where {N}
    ids = _resolve_subset(pool, curr_cands)
    new_pool, backmap = _restrict_pool(pool, ids)
    K = length(ids)
    inv = Dict{Int,PreferenceIndex}(Int(backmap[j]) => PreferenceIndex(j) for j in 1:K)
    keep = Set(Int.(ids))
    new_levels_ids = Vector{Vector{CandidateId}}()
    for lvl in wo.levels
        lvl2 = CandidateId[]
        for cid in lvl
            if Int(cid) in keep
                push!(lvl2, inv[Int(cid)])
            end
        end
        if !isempty(lvl2)
            push!(new_levels_ids, lvl2)
        end
    end
    new_wo = WeakOrder{K}(new_levels_ids)
    return new_wo, new_pool, backmap
end

"""
    restrict(pp::PairwisePreference, pool, curr_cands)
      -> (new_pp::PairwisePreference{Style,K}, new_pool, backmap)

Take the K×K principal submatrix induced by the subset (in the given order).
"""
function restrict(pp::PairwisePreference{Style,N,T},
                  pool::CandidatePool{N},
                  curr_cands) where {Style<:PreferenceStyle,N,T<:Integer}
    ids = _resolve_subset(pool, curr_cands)
    new_pool, backmap = _restrict_pool(pool, ids)
    K = length(ids)
    data = ntuple(i -> ntuple(j -> pp.matrix[Int(backmap[i]), Int(backmap[j])], Val(K)), Val(K))
    new_pp = PairwisePreference{Style,K,T}(SMatrix{K,K,T}(data))
    return new_pp, new_pool, backmap
end

# ---- subset helpers ----

"""
    _resolve_subset(pool, curr_cands) -> Vector{CandidateId}

Accepts ids or names; validates non-empty, no duplicates, id bounds.
"""
function _resolve_subset(pool::CandidatePool{N},
                         curr_cands) where {N}
    ids = CandidateId[]
    if eltype(curr_cands) <: Symbol
        for s in curr_cands
            push!(ids, pool[s])
        end
    else
        for x in curr_cands
            cid = CandidateId(x)
            (cid < 1 || cid > CandidateId(N)) && throw(ArgumentError("Candidate id $(x) out of bounds 1..$N"))
            push!(ids, cid)
        end
    end
    length(ids) == 0 && throw(ArgumentError("Restriction subset cannot be empty"))
    length(Set(ids)) == length(ids) || throw(ArgumentError("Restriction subset has duplicates"))
    return ids
end

"""
    _restrict_pool(pool, ids) -> (new_pool, backmap)

Make a new `CandidatePool` for the subset and a static `backmap` (new→old).
"""
function _restrict_pool(pool::CandidatePool{N},
                        ids::Vector{CandidateId}) where {N}
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

# ------------------------- asdict & CSV parsers -------------------------

"""
    asdict(pref::RankPreference, pool) -> Dict{Symbol,Int}

Emit `{name => rank}` for *present* ranks only (omits unranked/truncated).
"""
function asdict(pref::RankPreference{Style,N,R},
                pool::CandidatePool{N}) where {Style<:PreferenceStyle,N,R}
    d = Dict{Symbol,Int}()
    @inbounds for i in 1:N
        r = pref.ranks[i]
        if !ismissing(r)
            d[pool.names[i]] = Int(r)
        end
    end
    return d
end

# --- CSV ---

"""
    read_rank_columns_csv(path; candidates=nothing, style=:auto, missingstrings=["","","NA","NaN"])
      -> (ballots::Vector{RankPreference}, pool::CandidatePool)

Read a CSV where **each candidate is a column** and cells contain the voter's rank (or blank).

- If `candidates` is provided, validates columns exist and uses that order for the pool.
- If not, uses the file's columns in their file order.
- `style=:auto` infers Weak vs Strict per ballot from present ranks.
Requires `CSV.jl` at runtime.
"""
function read_rank_columns_csv(path::AbstractString;
                               candidates::Union{Nothing,Vector{Symbol}}=nothing,
                               style::Union{Symbol,Type}= :auto,
                               missingstrings = ["", "NA", "NaN"])
    CSV = _require_CSV()
    tbl = CSV.File(path; missingstring=missingstrings, silencewarnings=true)
    cols_all = Symbol.(propertynames(tbl))
    if candidates !== nothing
        missing_cols = setdiff(candidates, cols_all)
        !isempty(missing_cols) && throw(ArgumentError("Missing candidate columns: $(collect(missing_cols))"))
        cand_syms = candidates
    else
        cand_syms = cols_all
    end
    pool = CandidatePool(cand_syms)
    ballots = Vector{RankPreference}(undef, 0)
    for row in tbl
        d = Dict{Symbol,Int}()
        for c in cand_syms
            v = row[c]
            if v !== missing
                d[c] = Int(v)
            end
        end
        push!(ballots, RankPreference(pool, d; style=style))
    end
    return ballots, pool
end

"""
    read_candidate_columns_csv(path; cols, pool=nothing, style=:auto, missingstrings=["","","NA","NaN"])
      -> (ballots::Vector{RankPreference}, pool::CandidatePool)

Read a CSV where **columns encode ranked positions** (leftmost = top choice). Cells are names.

- `cols` must exist in the file (validated).
- If `pool` is omitted, infers it from the union of names (sorted for determinism).
- Ties are not representable in this simple variant; blanks imply truncation.
Requires `CSV.jl` at runtime.
"""
function read_candidate_columns_csv(path::AbstractString;
                                    cols::Vector{Symbol},
                                    pool::Union{Nothing,CandidatePool}=nothing,
                                    style::Union{Symbol,Type}= :auto,
                                    missingstrings = ["", "NA", "NaN"])
    CSV = _require_CSV()
    tbl = CSV.File(path; missingstring=missingstrings, silencewarnings=true)

    file_cols = Symbol.(propertynames(tbl))
    missing_cols = setdiff(cols, file_cols)
    !isempty(missing_cols) && throw(ArgumentError("Missing ordered columns: $(collect(missing_cols))"))

    # Infer pool if needed
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
        # re-open to iterate again
        tbl = CSV.File(path; missingstring=missingstrings, silencewarnings=true)
    else
        cp = pool
    end

    ballots = RankPreference[]
    for row in tbl
        ordered_syms = Symbol[]
        for c in cols
            v = row[c]
            if v !== missing
                s = v isa Symbol ? v : Symbol(String(v))
                push!(ordered_syms, s)
            end
        end
        if isempty(ordered_syms)
            # empty ballot → all missing ranks
            push!(ballots, RankPreference(cp, Dict{Symbol,Int}(); style=style))
        else
            ord = OrderPreference(cp, ordered_syms)
            push!(ballots, to_rank(Strict, cp, ord))
        end
    end
    return ballots, cp
end

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

# ------------------------- Exports -------------------------
# export CandidatePool, Strict, Weak,
#        RankPreference, OrderPreference, PairwisePreference, WeakOrder,
#        # Conversions:
#        to_rank, to_order, to_pairwise, to_weakorder,
#        # Predicates & utils:
#        rank, prefers, indifferent, ordered_candidates, tie_groups, asdict, linearize,
#        # Pairwise policies:
#        ext_bottom, ext_none,
#        # Weak-order helpers:
#        weakorder, weakorder_symbol_groups, ranks_from_weakorder,
#        # Restriction & convenience:
#        restrict, candidates, to_cmap, indifference_list,
#        # CSV readers:
#        read_rank_columns_csv, read_candidate_columns_csv









# # src/preferences_types.jl
# using StaticArrays: SVector, SMatrix

# const PreferenceIndex = UInt16
# const CandidateId = PreferenceIndex  # explicit alias for clarity when referring to ids

# """
#     CandidatePool{N}

# Immutable mapping between candidate *names* (symbols) and integer *ids* `1..N`.

# Design (pragmatic hybrid):
# - `names::SVector{N,Symbol}` stores the canonical, ordered list of candidates.
# - `index::Dict{Symbol,CandidateId}` maps name → id.

# Why this design?
# - Keeps `N` static for stack-allocated small arrays and good inference.
# - Avoids proliferating a distinct `CandidatePool{Names,N}` type per name set, reducing compilation and invalidations in exploratory workflows.

# Construction:
# - `CandidatePool(::AbstractVector{Symbol})`
# - `CandidatePool(::AbstractVector{<:AbstractString})`

# Conveniences:
# - `pool[sym]::CandidateId` (name→id)
# - `pool[i::Integer]::Symbol` (id→name)
# - `candidates(pool)::Vector{Symbol}`
# - `to_cmap(pool)::Dict{CandidateId,Symbol}`
# """
# struct CandidatePool{N}
#     names::SVector{N,Symbol}
#     index::Dict{Symbol,CandidateId}
# end

# """
#     CandidatePool(names::AbstractVector{Symbol})
#     CandidatePool(names::AbstractVector{<:AbstractString})

# Create a `CandidatePool` from unique candidate names. Throws if empty or duplicated.
# """
# function CandidatePool(names::AbstractVector{Symbol})
#     cand_tuple = Tuple(names)
#     N = length(cand_tuple)
#     N == 0 && throw(ArgumentError("Candidate pool cannot be empty"))
#     if length(Set(cand_tuple)) != N
#         throw(ArgumentError("Candidates must be unique"))
#     end
#     name_vec = SVector{N,Symbol}(cand_tuple)
#     # deterministic 1..N assignment in the canonical order
#     idx = Dict{Symbol,CandidateId}(s => CandidateId(i) for (i, s) in enumerate(name_vec))
#     CandidatePool{N}(name_vec, idx)
# end

# CandidatePool(names::AbstractVector{<:AbstractString}) =
#     CandidatePool(Symbol.(names))

# Base.length(pool::CandidatePool{N}) where {N} = N

# function Base.iterate(pool::CandidatePool{N}, state::Int=1) where {N}
#     state > N && return nothing
#     return pool.names[state], state + 1
# end

# """
#     pool[sym::Symbol] -> CandidateId

# Name → id lookup.
# """
# Base.getindex(pool::CandidatePool, cand::Symbol) = pool.index[cand]

# """
#     pool[i::Integer] -> Symbol

# Id → name lookup.
# """
# Base.getindex(pool::CandidatePool{N}, idx::Integer) where {N} = pool.names[idx]

# """
#     candidateat(pool, idx) -> Symbol

# Alias for `pool[idx]`.
# """
# candidateat(pool::CandidatePool{N}, idx::Integer) where {N} = pool.names[idx]

# """
#     keys(pool)

# Iterate candidate names in canonical order (returns the SVector of names).
# """
# Base.keys(pool::CandidatePool) = pool.names

# """
#     candidates(pool) -> Vector{Symbol}

# Materialize candidate names as a `Vector`.
# """
# candidates(pool::CandidatePool{N}) where {N} = collect(pool.names)

# """
#     to_cmap(pool) -> Dict{CandidateId,Symbol}

# Build an ID→name map compatible with common “cmap” conventions.
# """
# function to_cmap(pool::CandidatePool{N}) where {N}
#     d = Dict{CandidateId,Symbol}()
#     @inbounds for i in 1:N
#         d[CandidateId(i)] = pool.names[i]
#     end
#     return d
# end

# # ------------------------- Preferences core types -------------------------

# abstract type PreferenceStyle end
# """
# Tag for linear/strict orders (no ties)."""
# struct Strict <: PreferenceStyle end
# """
# Tag for weak orders (ties allowed)."""
# struct Weak   <: PreferenceStyle end

# abstract type AbstractPreference{Style<:PreferenceStyle,N} end

# """
#     RankPreference{Style,N,R<:Unsigned} <: AbstractPreference{Style,N}

# Ballot encoded as *ranks* per candidate id `1..N`. Lower rank = better.

# - `ranks::SVector{N,Union{Missing,R}}`:
#   - `missing` = unranked (truncation).
#   - Present ranks must be integers `≥ 1`. Ties allowed iff `Style === Weak`.

# Construct via helpers `RankPreference(pool, ...)` below.
# """
# struct RankPreference{Style<:PreferenceStyle,N,R<:Unsigned} <: AbstractPreference{Style,N}
#     ranks::SVector{N,Union{Missing,R}}  # missing = unranked
# end

# """
#     OrderPreference{Style,N,R<:Unsigned} <: AbstractPreference{Style,N}

# Ballot encoded as a *permutation* (strict order only).
# """
# struct OrderPreference{Style<:PreferenceStyle,N,R<:Unsigned} <: AbstractPreference{Style,N}
#     order::SVector{N,R}
# end

# """
#     PairwisePreference{Style,N,T<:Integer} <: AbstractPreference{Style,N}

# Pairwise comparison *sign* matrix at the ballot level:
# - entry `+1` if `i ≻ j`, `-1` if `j ≻ i`, `0` if tie/no-comparison (e.g., due to truncation policy).
# """
# struct PairwisePreference{Style<:PreferenceStyle,N,T<:Integer} <: AbstractPreference{Style,N}
#     matrix::SMatrix{N,N,T}
# end

# # ==================== WeakOrder (indifference classes) ====================

# """
#     WeakOrder{N}

# First-class weak order as levels of tied candidates.

# - `levels::Vector{Vector{CandidateId}}` with `levels[1]` the top indifference class, etc.
# - Every candidate `1..N` must appear *exactly once*.
# """
# struct WeakOrder{N}
#     levels::Vector{Vector{CandidateId}}
#     function WeakOrder{N}(levels::Vector{Vector{CandidateId}}) where {N}
#         seen = Set{CandidateId}()
#         for lvl in levels
#             for c in lvl
#                 (c < 1 || c > CandidateId(N)) && throw(ArgumentError("Candidate id $(c) out of 1..$N"))
#                 in(c, seen) && throw(ArgumentError("Candidate id $(c) appears in multiple levels"))
#                 push!(seen, c)
#             end
#         end
#         if length(seen) != N
#             throw(ArgumentError("WeakOrder must cover all $N candidates exactly once"))
#         end
#         new{N}(levels)
#     end
# end

# Base.length(::WeakOrder{N}) where {N} = N
# Base.iterate(wo::WeakOrder{N}, state::Int=1) where {N} = state > N ? nothing : (wo.levels[state], state+1)
# function Base.show(io::IO, wo::WeakOrder{N}) where {N}
#     print(io, "WeakOrder{$N}(", wo.levels, ")")
# end

# """
#     weakorder_symbol_groups(wo, pool) -> Vector{Vector{Symbol}}

# Render levels of a `WeakOrder` as candidate names.
# """
# function weakorder_symbol_groups(wo::WeakOrder{N}, pool::CandidatePool{N}) where {N}
#     [map(i -> pool.names[Int(i)], lvl) for lvl in wo.levels]
# end

# # ------------------------- Internal helpers -------------------------

# @inline _nonmissing(xs) = (x for x in xs if !ismissing(x))
# @inline function _min_pos_ignore_missing(xs)
#     m = typemax(Int)
#     found = false
#     for x in xs
#         if !ismissing(x)
#             found = true
#             m = min(m, Int(x))
#         end
#     end
#     return found ? m : nothing
# end

# """
#     _style_type(style) -> Type{Strict}|Type{Weak}

# Resolve user style input (`:strict`, `:weak`, `Strict`, `Weak`) to a concrete type.
# """
# @inline function _style_type(style)
#     style === :strict && return Strict
#     style === :weak   && return Weak
#     style === Strict  && return Strict
#     style === Weak    && return Weak
#     style <: PreferenceStyle && return style
#     throw(ArgumentError("Unknown preference style $(style)"))
# end

# """
#     _to_prefindex_or_missing(x) -> Union{Missing,PreferenceIndex}

# Safely convert a user rank value to `PreferenceIndex`, rejecting non-positive integers.
# `missing` is passed through unchanged.
# """
# @inline function _to_prefindex_or_missing(x)
#     ismissing(x) && return missing
#     xi = Int(x)
#     xi < 1 && throw(ArgumentError("Ranks must be ≥ 1, got $xi"))
#     return PreferenceIndex(xi)
# end

# """
#     _maybe_dict_rank(dict, cand) -> Union{Missing,PreferenceIndex}

# Read `cand` from a dictionary of ranks; missing key => `missing`.
# Values are validated via `_to_prefindex_or_missing`.
# """
# @inline function _maybe_dict_rank(dict::AbstractDict{Symbol,<:Integer}, cand::Symbol)
#     if haskey(dict, cand)
#         return _to_prefindex_or_missing(dict[cand])
#     else
#         return missing
#     end
# end

# """
#     _ensure_valid(::Type{Strict|Weak}, ranks_tuple)

# Validate a ranks tuple (with possible `missing`) according to the style.
# - Strict: all present ranks unique; all present ≥ 1.
# - Weak: all present ≥ 1.
# """
# @inline function _ensure_valid(::Type{Strict}, ranks::NTuple{N,Union{Missing,PreferenceIndex}}) where {N}
#     vals = collect(_nonmissing(ranks))
#     if length(Set(vals)) != length(vals)
#         throw(ArgumentError("Strict preferences require unique assigned ranks"))
#     end
#     m = _min_pos_ignore_missing(vals)
#     if m !== nothing && m < 1
#         throw(ArgumentError("Ranks must be ≥ 1"))
#     end
#     return nothing
# end

# @inline function _ensure_valid(::Type{Weak}, ranks::NTuple{N,Union{Missing,PreferenceIndex}}) where {N}
#     m = _min_pos_ignore_missing(ranks)
#     if m !== nothing && m < 1
#         throw(ArgumentError("Ranks must be ≥ 1"))
#     end
#     return nothing
# end

# """
#     _detect_style(ranks_tuple) -> Type{Strict|Weak}

# Infer style from present ranks: unique ⇒ `Strict`; duplicates ⇒ `Weak`.
# """
# @inline function _detect_style(ranks::NTuple{N,Union{Missing,PreferenceIndex}}) where {N}
#     vals = collect(_nonmissing(ranks))
#     return (length(Set(vals)) == length(vals)) ? Strict : Weak
# end

# """
#     _collect_ranks(pool, dict) -> NTuple{N,Union{Missing,PreferenceIndex}}

# Collect ranks for all candidates in `pool` from a dict `{name => rank}`.
# """
# @inline function _collect_ranks(pool::CandidatePool{N},
#                                 dict::AbstractDict{Symbol,<:Integer}) where {N}
#     ntuple(i -> _maybe_dict_rank(dict, pool.names[i]), Val(N))
# end

# # ------------------------- RankPreference constructors -------------------------

# """
#     RankPreference{Style}(pool, ranks_tuple)

# Construct from a tuple of validated `Union{Missing,PreferenceIndex}`.
# """
# RankPreference{Style}(pool::CandidatePool{N},
#                       ranks::NTuple{N,Union{Missing,PreferenceIndex}}) where {Style<:PreferenceStyle,N} =
#     ( _ensure_valid(Style, ranks);
#       RankPreference{Style,N,PreferenceIndex}(SVector{N,Union{Missing,PreferenceIndex}}(ranks)) )

# """
#     RankPreference{Style}(pool, ranks_tuple::NTuple{N,<:Integer})

# Construct from integer ranks; values are validated (`≥ 1`) then converted.
# """
# RankPreference{Style}(pool::CandidatePool{N},
#                       ranks::NTuple{N,<:Integer}) where {Style<:PreferenceStyle,N} =
#     RankPreference{Style}(pool, ntuple(i -> _to_prefindex_or_missing(ranks[i]), Val(N)))

# """
#     RankPreference{Style}(pool, ranks::SVector{N,Union{Missing,<:Integer}})

# Construct from an SVector with integers or `missing`.
# """
# RankPreference{Style}(pool::CandidatePool{N},
#                       ranks::SVector{N,Union{Missing,<:Integer}}) where {Style<:PreferenceStyle,N} =
#     RankPreference{Style}(pool, ntuple(i -> _to_prefindex_or_missing(ranks[i]), Val(N)))

# """
#     RankPreference{Style}(pool, ranks::SVector{N,<:Integer})  # no missing

# Construct from an SVector of integers.
# """
# RankPreference{Style}(pool::CandidatePool{N},
#                       ranks::SVector{N,<:Integer}) where {Style<:PreferenceStyle,N} =
#     RankPreference{Style}(pool, Tuple(ranks))

# """
#     RankPreference(pool, dict; style=:auto)

# Construct from `{name => rank}` dict. Missing keys ⇒ `missing` ranks.
# """
# function RankPreference(pool::CandidatePool{N},
#                         dict::AbstractDict{Symbol,<:Integer};
#                         style::Union{Symbol,Type}= :auto) where {N}
#     ranks = _collect_ranks(pool, dict)
#     Style = style === :auto ? _detect_style(ranks) : _style_type(style)
#     return RankPreference{Style}(pool, ranks)
# end

# """
#     RankPreference(pool, ranks::AbstractVector{<:Integer}; style=:auto)

# Construct from a dense vector of integer ranks (no truncation).
# """
# function RankPreference(pool::CandidatePool{N},
#                         ranks::AbstractVector{<:Integer};
#                         style::Union{Symbol,Type}= :auto) where {N}
#     length(ranks) == N || throw(ArgumentError("Expected $(N) ranks, got $(length(ranks))"))
#     ranks_tuple = ntuple(i -> _to_prefindex_or_missing(ranks[i]), Val(N))
#     Style = style === :auto ? _detect_style(ranks_tuple) : _style_type(style)
#     return RankPreference{Style}(pool, ranks_tuple)
# end

# """
#     RankPreference(pool, ranks::AbstractVector{T}; style=:auto)

# Construct from a vector that may contain `missing`. Non-missing validated to be ≥ 1.
# """
# function RankPreference(pool::CandidatePool{N},
#                         ranks::AbstractVector{T};
#                         style::Union{Symbol,Type}= :auto) where {N,T}
#     length(ranks) == N || throw(ArgumentError("Expected $(N) ranks, got $(length(ranks))"))
#     ranks_tuple = ntuple(i -> _to_prefindex_or_missing(ranks[i]), Val(N))
#     Style = style === :auto ? _detect_style(ranks_tuple) : _style_type(style)
#     return RankPreference{Style}(pool, ranks_tuple)
# end

# """
#     OrderPreference(pool, order_names)

# Strict order constructed from a full permutation of candidate names.
# """
# function OrderPreference(pool::CandidatePool{N},
#                          order::AbstractVector{Symbol}) where {N}
#     length(order) == N || throw(ArgumentError("Expected $(N) candidates, got $(length(order))"))
#     idxs = ntuple(i -> PreferenceIndex(pool[order[i]]), Val(N))
#     length(Set(idxs)) == N || throw(ArgumentError("Order must mention each candidate once"))
#     return OrderPreference{Strict,N,PreferenceIndex}(SVector{N,PreferenceIndex}(idxs))
# end

# """
#     OrderPreference(pref::RankPreference{Strict}, pool)

# Derive a strict order from strict ranks (ties absent). Ranked first by `(rank)`, then index.
# Unranked entries (shouldn't occur for strict) are placed last defensively.
# """
# function OrderPreference(pref::RankPreference{Strict,N,R},
#                          pool::CandidatePool{N}) where {N,R}
#     perm = sortperm(collect(1:N),
#                     by = i -> begin
#                         r = pref.ranks[i]
#                         ismissing(r) ? (1, typemax(Int), i) : (0, Int(r), i)
#                     end)
#     idxs = ntuple(i -> PreferenceIndex(perm[i]), Val(N))
#     return OrderPreference{Strict,N,PreferenceIndex}(SVector{N,PreferenceIndex}(idxs))
# end

# # ================= Pairwise extraction with extension policy =================

# """
#     ext_bottom(ra, rb, i, j, ranks, pool) -> Int

# Default extension: ranked ≻ unranked. Both unranked ⇒ tie (0).
# """
# @inline function ext_bottom(ra, rb, i, j, ranks, pool)
#     if ismissing(ra) && ismissing(rb)
#         return 0
#     elseif ismissing(ra)
#         return -1
#     elseif ismissing(rb)
#         return 1
#     else
#         d = Int(rb) - Int(ra) # lower rank = better
#         return d > 0 ? 1 : d < 0 ? -1 : 0
#     end
# end

# """
#     ext_none(ra, rb, i, j, ranks, pool) -> Int

# Strict-only comparisons: any `missing` ⇒ no-comparison (0).
# """
# @inline function ext_none(ra, rb, i, j, ranks, pool)
#     if ismissing(ra) || ismissing(rb)
#         return 0
#     else
#         d = Int(rb) - Int(ra)
#         return d > 0 ? 1 : d < 0 ? -1 : 0
#     end
# end

# """
#     _resolve_extension(extension) -> Function

# Accepts `:bottom`, `:none`, or a custom `extension(ra,rb,i,j,ranks,pool)::Int`.
# """
# @inline function _resolve_extension(extension)
#     if extension === :bottom || extension === nothing
#         return ext_bottom
#     elseif extension === :none
#         return ext_none
#     elseif extension isa Function
#         return extension
#     else
#         throw(ArgumentError("Unknown extension method $(extension). Use :bottom, :none, or a function."))
#     end
# end

# """
#     PairwisePreference(pref, pool; extension=:bottom) -> PairwisePreference

# Build pairwise sign matrix for a single ballot using the chosen extension/imputation policy.
# """
# function PairwisePreference(pref::RankPreference{Style,N,R},
#                             pool::CandidatePool{N};
#                             extension = :bottom) where {Style<:PreferenceStyle,N,R}
#     ext = _resolve_extension(extension)
#     data = ntuple(i -> ntuple(j -> _pairwise_entry(pref.ranks, i, j, ext, pool), Val(N)), Val(N))
#     return PairwisePreference{Style,N,Int8}(SMatrix{N,N,Int8}(data))
# end

# @inline function _pairwise_entry(ranks::SVector{N,Union{Missing,R}},
#                                  i::Int, j::Int, ext, pool) where {N,R}
#     i == j && return Int8(0)
#     ai = ranks[i]; bj = ranks[j]
#     return Int8(ext(ai, bj, i, j, ranks, pool))
# end

# # ------------------------- Predicates & utilities -------------------------

# """
#     rank(pref, pool, name::Symbol) -> Union{Missing,PreferenceIndex}

# Lookup a candidate’s rank in a ballot (may be `missing`).
# """
# function rank(pref::RankPreference{Style,N,R},
#               pool::CandidatePool{N},
#               cand::Symbol) where {Style<:PreferenceStyle,N,R}
#     return pref.ranks[pool[cand]]
# end

# """
#     prefers(pref, pool, a::Symbol, b::Symbol) -> Bool

# True iff both ranked and `a` has a strictly smaller rank than `b`.
# """
# function prefers(pref::RankPreference, pool::CandidatePool, a::Symbol, b::Symbol)
#     ra = rank(pref, pool, a); rb = rank(pref, pool, b)
#     (ismissing(ra) || ismissing(rb)) && return false
#     return ra < rb
# end

# """
#     indifferent(pref, pool, a::Symbol, b::Symbol) -> Bool

# True iff both ranked and tied.
# """
# function indifferent(pref::RankPreference, pool::CandidatePool, a::Symbol, b::Symbol)
#     ra = rank(pref, pool, a); rb = rank(pref, pool, b)
#     (ismissing(ra) || ismissing(rb)) && return false
#     return ra == rb
# end

# """
#     ordered_candidates(pref::RankPreference{Strict}, pool) -> Vector{Symbol}

# Return names sorted by strict rank; unranked (defensive) at the end.
# """
# function ordered_candidates(pref::RankPreference{Strict,N,R},
#                             pool::CandidatePool{N}) where {N,R}
#     perm = sortperm(collect(1:N),
#                     by = i -> begin
#                         r = pref.ranks[i]
#                         ismissing(r) ? (1, typemax(Int), i) : (0, Int(r), i)
#                     end)
#     return collect(pool.names[perm])
# end

# # ------------------------- WeakOrder construction & conversions -------------------------

# """
#     weakorder(pref::RankPreference{Weak}) -> WeakOrder

# Convert weak ranks (with truncation) into an explicit levels representation.
# Unranked candidates are placed in the last level so coverage is total.
# """
# function weakorder(pref::RankPreference{Weak,N,R}) where {N,R}
#     groups = Dict{PreferenceIndex,Vector{CandidateId}}()
#     unranked = CandidateId[]
#     @inbounds for i in 1:N
#         r = pref.ranks[i]
#         if ismissing(r)
#             push!(unranked, CandidateId(i))
#         else
#             push!(get!(groups, PreferenceIndex(r), CandidateId[]), CandidateId(i))
#         end
#     end
#     ranked_levels = [groups[k] for k in sort!(collect(keys(groups)))]
#     levels = isempty(unranked) ? ranked_levels : vcat(ranked_levels, [unranked])
#     return WeakOrder{N}(levels)
# end

# """
#     indifference_list(pref::RankPreference{Weak}) -> Vector{Vector{CandidateId}}

# Lossless ID-based indifference classes (levels). Use `weakorder_symbol_groups` to view names.
# """
# function indifference_list(pref::RankPreference{Weak,N,R}) where {N,R}
#     return weakorder(pref).levels
# end

# """
#     tie_groups(pref::RankPreference{Weak}, pool) -> Vector{Vector{Symbol}}

# Backward-compatible symbol rendering for ties.
# """
# function tie_groups(pref::RankPreference{Weak,N,R},
#                     pool::CandidatePool{N}) where {N,R}
#     wo = weakorder(pref)
#     return weakorder_symbol_groups(wo, pool)
# end

# """
#     ranks_from_weakorder(wo) -> NTuple{N,PreferenceIndex}

# Assign equal rank to all ids within the same level, level order = rank order.
# """
# function ranks_from_weakorder(wo::WeakOrder{N}) where {N}
#     ranks = Vector{PreferenceIndex}(undef, N)
#     @inbounds for (lvl, ids) in enumerate(wo.levels)
#         for cid in ids
#             ranks[Int(cid)] = PreferenceIndex(lvl)
#         end
#     end
#     return ntuple(i -> ranks[i], Val(N))
# end

# """
#     linearize(wo::WeakOrder) -> NTuple{N,PreferenceIndex}

# Produce strict ranks by ordering levels then IDs within each level.
# """
# function linearize(wo::WeakOrder{N}) where {N}
#     ranks = Vector{PreferenceIndex}(undef, N)
#     pos = 1
#     @inbounds for ids in wo.levels
#         for cid in sort(ids)
#             ranks[Int(cid)] = PreferenceIndex(pos)
#             pos += 1
#         end
#     end
#     return ntuple(i -> ranks[i], Val(N))
# end

# """
#     linearize(pref::RankPreference{Weak}, pool) -> RankPreference{Strict}

# Tie-break by `(rank, id)` then assign 1..N.
# """
# function linearize(pref::RankPreference{Weak,N,R},
#                    pool::CandidatePool{N}) where {N,R}
#     order = collect(1:N)
#     sort!(order, by = i -> (ismissing(pref.ranks[i]) ? (true,  typemax(Int), i)
#                                                      : (false, Int(pref.ranks[i]), i)))
#     ranks = Vector{PreferenceIndex}(undef, N)
#     @inbounds for (pos, idx) in enumerate(order)
#         ranks[idx] = PreferenceIndex(pos)
#     end
#     return RankPreference{Strict}(pool, ntuple(i -> ranks[i], Val(N)))
# end

# # ------------------------- Converters -------------------------

# """
#     ToRank{Style}(pool)

# Functor converting inputs (dicts, orders, weak orders, rank preferences) into `RankPreference{Style}`.
# """
# struct ToRank{Style<:PreferenceStyle}
#     pool::CandidatePool
# end

# (f::ToRank{Style})(pref::RankPreference{Style}) where Style = pref

# """
#     (ToRank{Style}(pool))(dict::Dict{Symbol,Int}; style_override?)

# Dict → ranks. Missing keys become `missing` ranks.
# """
# function (f::ToRank{Style})(dict::AbstractDict{Symbol,<:Integer}) where Style
#     return RankPreference(f.pool, dict; style=Style)
# end

# """
#     (ToRank{Weak}|ToRank{Strict})(::RankPreference{Strict|Weak})
#     (ToRank{Weak|Strict})(::WeakOrder)
#     (ToRank{Style})(::OrderPreference{Strict})

# Overloaded conversions among ballot encodings.
# """
# function (f::ToRank{Weak})(pref::RankPreference{Strict})
#     return RankPreference{Weak}(f.pool, Tuple(pref.ranks))
# end
# function (f::ToRank{Strict})(pref::RankPreference{Weak})
#     return linearize(pref, f.pool)
# end
# function (f::ToRank{Weak})(wo::WeakOrder{N}) where {N}
#     ranks_tuple = ranks_from_weakorder(wo)
#     return RankPreference{Weak}(f.pool, ranks_tuple)
# end
# function (f::ToRank{Strict})(wo::WeakOrder{N}) where {N}
#     ranks_tuple = linearize(wo)
#     return RankPreference{Strict}(f.pool, ranks_tuple)
# end
# function (f::ToRank{Style})(order::OrderPreference{Strict}) where Style
#     ranks = _order_to_rank_tuple(order)
#     return RankPreference{Style}(f.pool, ranks)
# end

# @inline function _order_to_rank_tuple(order::OrderPreference{Strict,N,R}) where {N,R}
#     inv = Vector{PreferenceIndex}(undef, N)
#     @inbounds for (pos, idx) in enumerate(order.order)
#         inv[Int(idx)] = PreferenceIndex(pos)
#     end
#     return ntuple(i -> inv[i], Val(N))
# end

# """
#     ToOrder{Style}(pool)

# Functor converting preferences into `OrderPreference{Strict}` or `WeakOrder`.
# - `ToOrder{Strict}` yields a strict permutation.
# - `ToOrder{Weak}` yields a `WeakOrder`.
# """
# struct ToOrder{Style<:PreferenceStyle}
#     pool::CandidatePool
# end

# (f::ToOrder{Strict})(order::OrderPreference{Strict}) = order

# """
#     (ToOrder{Strict})(::RankPreference{Strict}|::Dict)

# Rank/dict → strict order.
# """
# function (f::ToOrder{Strict})(pref::RankPreference{Strict})
#     return OrderPreference(pref, f.pool)
# end
# function (f::ToOrder{Strict})(dict::AbstractDict{Symbol,<:Integer})
#     pref = (ToRank{Strict}(f.pool))(dict)
#     return OrderPreference(pref, f.pool)
# end

# """
#     (ToOrder{Weak})(::RankPreference{Weak}|::Dict) -> WeakOrder
# """
# function (f::ToOrder{Weak})(pref::RankPreference{Weak,N,R}) where {N,R}
#     return weakorder(pref)
# end
# function (f::ToOrder{Weak})(dict::AbstractDict{Symbol,<:Integer})
#     pref = (ToRank{Weak}(f.pool))(dict)
#     return weakorder(pref)
# end

# """
#     ToPairwise{Style}(pool)

# Functor converting ballots to `PairwisePreference{Style}` using an extension policy.
# """
# struct ToPairwise{Style<:PreferenceStyle}
#     pool::CandidatePool
# end

# """
#     (ToPairwise{Style})(pref|dict|order|weakorder; extension=:bottom) -> PairwisePreference

# Build ballot-level pairwise sign matrix using the chosen policy.
# """
# function (f::ToPairwise{Style})(pref::RankPreference{Style};
#                                 extension = :bottom) where Style
#     return PairwisePreference(pref, f.pool; extension=extension)
# end
# function (f::ToPairwise{Style})(dict::AbstractDict{Symbol,<:Integer};
#                                 extension = :bottom) where Style
#     pref = (ToRank{Style}(f.pool))(dict)
#     return PairwisePreference(pref, f.pool; extension=extension)
# end
# function (f::ToPairwise{Style})(order::OrderPreference{Strict};
#                                 extension = :bottom) where Style
#     pref = (ToRank{Style}(f.pool))(order)
#     return PairwisePreference(pref, f.pool; extension=extension)
# end
# function (f::ToPairwise{Style})(wo::WeakOrder{N};
#                                 extension = :bottom) where {Style,N}
#     pref = (ToRank{Style}(f.pool))(wo)
#     return PairwisePreference(pref, f.pool; extension=extension)
# end

# # ------------------------- Restriction hooks -------------------------

# """
#     restrict(pool, curr_cands) -> (new_pool, backmap)

# Restrict a `CandidatePool` to `curr_cands` (ids or names), preserving order.
# Returns a new pool and a `backmap::SVector{K,CandidateId}` (new id → old id).
# """
# function restrict(pool::CandidatePool{N}, curr_cands) where {N}
#     ids = _resolve_subset(pool, curr_cands)
#     _restrict_pool(pool, ids)
# end

# """
#     restrict(pref::RankPreference, pool, curr_cands)
#       -> (new_pref::RankPreference{Style,K}, new_pool, backmap)

# Restrict ranks to a candidate subset. Present ranks are re-compressed to contiguous
# `1..L` preserving ties; `missing` remains `missing`.
# """
# function restrict(pref::RankPreference{Style,N,R},
#                   pool::CandidatePool{N},
#                   curr_cands) where {Style<:PreferenceStyle,N,R}
#     ids = _resolve_subset(pool, curr_cands)
#     new_pool, backmap = _restrict_pool(pool, ids)
#     K = length(ids)
#     sel = Vector{Union{Missing,PreferenceIndex}}(undef, K)
#     @inbounds for j in 1:K
#         sel[j] = pref.ranks[Int(backmap[j])]
#     end
#     selc = _compress_ranks(sel)
#     new_ranks = ntuple(j -> selc[j], Val(K))
#     new_pref = RankPreference{Style}(new_pool, new_ranks)
#     return new_pref, new_pool, backmap
# end

# """
#     restrict(order::OrderPreference{Strict}, pool, curr_cands)
#       -> (new_order::OrderPreference{Strict,K}, new_pool, backmap)

# Restrict a strict order to a subset, preserving original relative order.
# """
# function restrict(order::OrderPreference{Strict,N,R},
#                   pool::CandidatePool{N},
#                   curr_cands) where {N,R}
#     ids = _resolve_subset(pool, curr_cands)
#     new_pool, backmap = _restrict_pool(pool, ids)
#     K = length(ids)
#     orig = collect(Int.(order.order))
#     keep = Set(Int.(ids))
#     filtered = Int[]
#     @inbounds for idx in orig
#         if idx in keep
#             push!(filtered, idx)
#         end
#     end
#     inv = Dict{Int,PreferenceIndex}(Int(backmap[j]) => PreferenceIndex(j) for j in 1:K)
#     new_order_vec = [inv[i] for i in filtered]
#     new_order = OrderPreference{Strict,K,PreferenceIndex}(SVector{K,PreferenceIndex}(Tuple(new_order_vec)))
#     return new_order, new_pool, backmap
# end

# """
#     restrict(wo::WeakOrder, pool, curr_cands)
#       -> (new_wo::WeakOrder{K}, new_pool, backmap)

# Filter each level to the subset, drop empties, and reindex ids to `1..K`.
# """
# function restrict(wo::WeakOrder{N},
#                   pool::CandidatePool{N},
#                   curr_cands) where {N}
#     ids = _resolve_subset(pool, curr_cands)
#     new_pool, backmap = _restrict_pool(pool, ids)
#     K = length(ids)
#     inv = Dict{Int,PreferenceIndex}(Int(backmap[j]) => PreferenceIndex(j) for j in 1:K)
#     keep = Set(Int.(ids))
#     new_levels_ids = Vector{Vector{CandidateId}}()
#     for lvl in wo.levels
#         lvl2 = CandidateId[]
#         for cid in lvl
#             if Int(cid) in keep
#                 push!(lvl2, inv[Int(cid)])
#             end
#         end
#         if !isempty(lvl2)
#             push!(new_levels_ids, lvl2)
#         end
#     end
#     new_wo = WeakOrder{K}(new_levels_ids)
#     return new_wo, new_pool, backmap
# end

# """
#     restrict(pp::PairwisePreference, pool, curr_cands)
#       -> (new_pp::PairwisePreference{Style,K}, new_pool, backmap)

# Take the K×K principal submatrix induced by the subset (in the given order).
# """
# function restrict(pp::PairwisePreference{Style,N,T},
#                   pool::CandidatePool{N},
#                   curr_cands) where {Style<:PreferenceStyle,N,T<:Integer}
#     ids = _resolve_subset(pool, curr_cands)
#     new_pool, backmap = _restrict_pool(pool, ids)
#     K = length(ids)
#     data = ntuple(i -> ntuple(j -> pp.matrix[Int(backmap[i]), Int(backmap[j])], Val(K)), Val(K))
#     new_pp = PairwisePreference{Style,K,T}(SMatrix{K,K,T}(data))
#     return new_pp, new_pool, backmap
# end

# # ---- subset helpers ----

# """
#     _resolve_subset(pool, curr_cands) -> Vector{CandidateId}

# Accepts ids or names; validates non-empty, no duplicates, id bounds.
# """
# function _resolve_subset(pool::CandidatePool{N},
#                          curr_cands) where {N}
#     ids = CandidateId[]
#     if eltype(curr_cands) <: Symbol
#         for s in curr_cands
#             push!(ids, pool[s])
#         end
#     else
#         for x in curr_cands
#             cid = CandidateId(x)
#             (cid < 1 || cid > CandidateId(N)) && throw(ArgumentError("Candidate id $(x) out of bounds 1..$N"))
#             push!(ids, cid)
#         end
#     end
#     length(ids) == 0 && throw(ArgumentError("Restriction subset cannot be empty"))
#     length(Set(ids)) == length(ids) || throw(ArgumentError("Restriction subset has duplicates"))
#     return ids
# end

# """
#     _restrict_pool(pool, ids) -> (new_pool, backmap)

# Make a new `CandidatePool` for the subset and a static `backmap` (new→old).
# """
# function _restrict_pool(pool::CandidatePool{N},
#                         ids::Vector{CandidateId}) where {N}
#     K = length(ids)
#     new_names = [pool.names[Int(i)] for i in ids]
#     new_pool = CandidatePool(new_names)
#     backmap = SVector{K,CandidateId}(Tuple(ids))
#     return new_pool, backmap
# end

# """
#     _compress_ranks(selected::Vector{Union{Missing,PreferenceIndex}}) -> Vector{Union{Missing,PreferenceIndex}}

# Map present ranks to `1..L` preserving ties; keep `missing` as `missing`.
# """
# function _compress_ranks(selected::Vector{Union{Missing,PreferenceIndex}})
#     present = unique(filter(!ismissing, selected))
#     sort!(present)
#     m = Dict{PreferenceIndex,PreferenceIndex}(r => PreferenceIndex(i) for (i,r) in enumerate(present))
#     out = Vector{Union{Missing,PreferenceIndex}}(undef, length(selected))
#     @inbounds for i in eachindex(selected)
#         r = selected[i]
#         out[i] = ismissing(r) ? missing : m[PreferenceIndex(r)]
#     end
#     return out
# end

# # ------------------------- asdict & CSV parsers -------------------------

# """
#     asdict(pref::RankPreference, pool) -> Dict{Symbol,Int}

# Emit `{name => rank}` for *present* ranks only (omits unranked/truncated).
# """
# function asdict(pref::RankPreference{Style,N,R},
#                 pool::CandidatePool{N}) where {Style<:PreferenceStyle,N,R}
#     d = Dict{Symbol,Int}()
#     @inbounds for i in 1:N
#         r = pref.ranks[i]
#         if !ismissing(r)
#             d[pool.names[i]] = Int(r)
#         end
#     end
#     return d
# end

# # --- CSV ---

# """
#     read_rank_columns_csv(path; candidates=nothing, style=:auto, missingstrings=["","","NA","NaN"])
#       -> (ballots::Vector{RankPreference}, pool::CandidatePool)

# Read a CSV where **each candidate is a column** and cells contain the voter's rank (or blank).

# - If `candidates` is provided, validates columns exist and uses that order for the pool.
# - If not, uses the file's columns in their file order.
# - `style=:auto` infers Weak vs Strict per ballot from present ranks.
# Requires `CSV.jl` at runtime.
# """
# function read_rank_columns_csv(path::AbstractString;
#                                candidates::Union{Nothing,Vector{Symbol}}=nothing,
#                                style::Union{Symbol,Type}= :auto,
#                                missingstrings = ["", "NA", "NaN"])
#     CSV = _require_CSV()
#     tbl = CSV.File(path; missingstring=missingstrings, silencewarnings=true)
#     cols_all = Symbol.(propertynames(tbl))
#     if candidates !== nothing
#         missing_cols = setdiff(candidates, cols_all)
#         !isempty(missing_cols) && throw(ArgumentError("Missing candidate columns: $(collect(missing_cols))"))
#         cand_syms = candidates
#     else
#         cand_syms = cols_all
#     end
#     pool = CandidatePool(cand_syms)
#     ballots = Vector{RankPreference}(undef, 0)
#     for row in tbl
#         d = Dict{Symbol,Int}()
#         for c in cand_syms
#             v = row[c]
#             if v !== missing
#                 d[c] = Int(v)
#             end
#         end
#         push!(ballots, RankPreference(pool, d; style=style))
#     end
#     return ballots, pool
# end

# """
#     read_candidate_columns_csv(path; cols, pool=nothing, style=:auto, missingstrings=["","","NA","NaN"])
#       -> (ballots::Vector{RankPreference}, pool::CandidatePool)

# Read a CSV where **columns encode ranked positions** (leftmost = top choice). Cells are names.

# - `cols` must exist in the file (validated).
# - If `pool` is omitted, infers it from the union of names (sorted for determinism).
# - Ties are not representable in this simple variant; blanks imply truncation.
# Requires `CSV.jl` at runtime.
# """
# function read_candidate_columns_csv(path::AbstractString;
#                                     cols::Vector{Symbol},
#                                     pool::Union{Nothing,CandidatePool}=nothing,
#                                     style::Union{Symbol,Type}= :auto,
#                                     missingstrings = ["", "NA", "NaN"])
#     CSV = _require_CSV()
#     tbl = CSV.File(path; missingstring=missingstrings, silencewarnings=true)

#     file_cols = Symbol.(propertynames(tbl))
#     missing_cols = setdiff(cols, file_cols)
#     !isempty(missing_cols) && throw(ArgumentError("Missing ordered columns: $(collect(missing_cols))"))

#     # Infer pool if needed
#     local cp
#     if pool === nothing
#         seen = Set{Symbol}()
#         for row in tbl
#             for c in cols
#                 v = row[c]
#                 if v !== missing
#                     s = v isa Symbol ? v : Symbol(String(v))
#                     push!(seen, s)
#                 end
#             end
#         end
#         cp = CandidatePool(sort!(collect(seen)))
#         # re-open to iterate again
#         tbl = CSV.File(path; missingstring=missingstrings, silencewarnings=true)
#     else
#         cp = pool
#     end

#     ballots = RankPreference[]
#     TRS = ToRank{Strict}(cp)
#     for row in tbl
#         ordered_syms = Symbol[]
#         for c in cols
#             v = row[c]
#             if v !== missing
#                 s = v isa Symbol ? v : Symbol(String(v))
#                 push!(ordered_syms, s)
#             end
#         end
#         if isempty(ordered_syms)
#             push!(ballots, RankPreference(cp, Dict{Symbol,Int}(); style=style))
#         else
#             ord = OrderPreference(cp, ordered_syms)
#             push!(ballots, TRS(ord))
#         end
#     end
#     return ballots, cp
# end

# """
#     _require_CSV() -> Module

# Lazy-import CSV.jl or throw a helpful error.
# """
# @inline function _require_CSV()
#     try
#         @eval import CSV
#         return CSV
#     catch
#         throw(ArgumentError("read_*_csv requires CSV.jl. Please: using Pkg; Pkg.add(\"CSV\"); then retry."))
#     end
# end

# # # ------------------------- Exports -------------------------

# # export CandidatePool, Strict, Weak,
# #        RankPreference, OrderPreference, PairwisePreference, WeakOrder,
# #        ToRank, ToOrder, ToPairwise,
# #        rank, prefers, indifferent,
# #        ordered_candidates, tie_groups,
# #        asdict, linearize,
# #        ext_bottom, ext_none,
# #        weakorder, weakorder_symbol_groups, ranks_from_weakorder,
# #        restrict, candidates, to_cmap,
# #        indifference_list,
# #        read_rank_columns_csv, read_candidate_columns_csv
