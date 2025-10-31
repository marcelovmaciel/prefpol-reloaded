#### preferences_types.jl — Core types & algorithms (no UI deps)


# Requires preferences_ext.jl to be included first.
# Uses: ExtensionPolicy, BottomPolicyMissing, NonePolicyMissing, resolve_policy, compare_maybe

const PreferenceIndex = UInt16
const CandidateId     = PreferenceIndex

# ============================ CandidatePool =============================
"""
CandidatePool{N}

Immutable mapping between candidate names (Symbols) and integer ids `1..N`.

- `names::SVector{N,Symbol}`: canonical ordered list of candidates
- `index::Dict{Symbol,CandidateId}`: name → id
"""
struct CandidatePool{N}
    names::SVector{N,Symbol}
    index::Dict{Symbol,CandidateId}  # NOTE: Dict is mutable; treat as private.
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

candidateat(pool::CandidatePool{N}, idx::Integer) where {N} = pool.names[idx]
Base.keys(pool::CandidatePool) = pool.names
candidates(pool::CandidatePool{N}) where {N} = collect(pool.names)

function to_cmap(pool::CandidatePool{N}) where {N}
    d = Dict{CandidateId,Symbol}()
    @inbounds for i in 1:N
        d[CandidateId(i)] = pool.names[i]
    end
    return d
end

# ============================ Preference types ===========================
abstract type AbstractPreference{N} end

"Rank-encoded ballot allowing ties and truncation."
struct WeakRank{N} <: AbstractPreference{N}
    ranks::SVector{N,Union{Missing,PreferenceIndex}}
end

"Strict total order encoded as contiguous ranks `1..N` with no `missing` and no ties."
struct StrictRank{N} <: AbstractPreference{N}
    ranks::SVector{N,PreferenceIndex}
    function StrictRank{N}(r::SVector{N,PreferenceIndex}) where {N}
        # Validate 1..N, unique
        seen = falses(N); mn = typemax(Int); mx = typemin(Int)
        @inbounds for k in r
            i = Int(k)
            (1 ≤ i ≤ N) || throw(ArgumentError("Strict ranks must be in 1..$N, got $i"))
            seen[i] && throw(ArgumentError("Strict ranks must be unique; duplicate $i"))
            seen[i] = true
            mn = min(mn, i); mx = max(mx, i)
        end
        (mn == 1 && mx == N) || throw(ArgumentError("Strict ranks must cover 1..$N"))
        return new{N}(r)
    end
end

"Ballot-level pairwise comparison matrix with skew-symmetry and 0 diagonal."
struct Pairwise{N,T} <: AbstractPreference{N}
    matrix::SMatrix{N,N,T}
    function Pairwise{N,T}(M::SMatrix{N,N,T}) where {N,T}
        @inbounds for i in 1:N
            ismissing(M[i,i]) && throw(ArgumentError("Diagonal cannot be missing at ($i,$i)"))
            M[i,i] == 0 || throw(ArgumentError("Diagonal must be 0 at ($i,$i), got $(M[i,i])"))
            for j in i+1:N
                a = M[i,j]; b = M[j,i]
                if ismissing(a) || ismissing(b)
                    (ismissing(a) && ismissing(b)) || throw(ArgumentError("Missing must be symmetric at ($i,$j) and ($j,$i)"))
                else
                    (a == -1 || a == 0 || a == 1) || throw(ArgumentError("Invalid entry $a at ($i,$j); use -1, 0, 1, or missing"))
                    (b == -1 || b == 0 || b == 1) || throw(ArgumentError("Invalid entry $b at ($j,$i); use -1, 0, 1, or missing"))
                    a == -b || throw(ArgumentError("Skew-symmetry violated at ($i,$j): $a vs $b"))
                end
            end
        end
        return new{N,T}(M)
    end
end

# ============================ Helpers ====================================
@inline _nonmissing(xs) = (x for x in xs if !ismissing(x))

@inline function _to_prefindex_or_missing(x)
    ismissing(x) && return missing
    if x isa Integer
        xi = Int(x)
    else
        isinteger(x) || throw(ArgumentError("Ranks must be integers; got $x"))
        xi = Int(x)
    end
    xi < 1 && throw(ArgumentError("Ranks must be ≥ 1, got $xi"))
    return PreferenceIndex(xi)
end

@inline function _collect_ranks(pool::CandidatePool{N}, dict::AbstractDict{Symbol,<:Integer}) where {N}
    ntuple(i -> begin
        s = pool.names[i]
        haskey(dict, s) ? _to_prefindex_or_missing(dict[s]) : missing
    end, Val(N))
end

# ============================ Constructors ===============================
function WeakRank(pool::CandidatePool{N}, dict::AbstractDict{Symbol,<:Integer}) where {N}
    ranks = _collect_ranks(pool, dict)
    return WeakRank{N}(SVector{N,Union{Missing,PreferenceIndex}}(ranks))
end
function WeakRank(pool::CandidatePool{N}, v::AbstractVector{T}) where {N,T}
    length(v) == N || throw(ArgumentError("Expected $(N) ranks, got $(length(v))"))
    ranks = ntuple(i -> _to_prefindex_or_missing(v[i]), Val(N))
    return WeakRank{N}(SVector{N,Union{Missing,PreferenceIndex}}(ranks))
end

function StrictRank(pool::CandidatePool{N}, v::AbstractVector{<:Integer}) where {N}
    length(v) == N || throw(ArgumentError("Expected $(N) ranks, got $(length(v))"))
    tu = ntuple(i -> PreferenceIndex(v[i]), Val(N))
    return StrictRank{N}(SVector{N,PreferenceIndex}(tu))
end

function StrictRank(pool::CandidatePool{N}, order::AbstractVector{<:Symbol}) where {N}
    length(order) == N || throw(ArgumentError("Expected $(N) candidates, got $(length(order))"))
    seen = Set{Symbol}(); ids = Vector{Int}(undef, N)
    @inbounds for (pos, s) in enumerate(order)
        haskey(pool.index, s) || throw(ArgumentError("Unknown candidate: $(s)"))
        in(s, seen) && throw(ArgumentError("Duplicate candidate: $(s)"))
        push!(seen, s)
        ids[pos] = Int(pool[s])
    end
    length(seen) == N || throw(ArgumentError("Permutation must include all $(N) candidates exactly once"))
    ranks = Vector{PreferenceIndex}(undef, N)
    @inbounds for (pos, cid_int) in enumerate(ids)
        ranks[cid_int] = PreferenceIndex(pos)
    end
    return StrictRank{N}(SVector{N,PreferenceIndex}(Tuple(ranks)))
end

# ============================ Views / Conversions =========================
to_weak(x::StrictRank{N}) where {N} =
    WeakRank{N}(SVector{N,Union{Missing,PreferenceIndex}}(ntuple(i->x.ranks[i], Val(N))))

function to_strict(x::WeakRank{N}; tie_break=:error, pool::Union{Nothing,CandidatePool{N}}=nothing) where {N}
    r = x.ranks
    if tie_break === :error
        all(!ismissing, r) || throw(ArgumentError("Cannot strictify: missing ranks present"))
        vals = collect(r)
        length(Set(vals)) == N || throw(ArgumentError("Cannot strictify: ties present"))
        tu = ntuple(i->PreferenceIndex(Int(vals[i])), Val(N))
        return StrictRank{N}(SVector{N,PreferenceIndex}(tu))
    elseif tie_break isa Function
        f = tie_break
        order = collect(1:N)
        sort!(order; lt = (i, j) -> f(i, j, r, pool))
        ranks = Vector{PreferenceIndex}(undef, N)
        @inbounds for (pos, idx) in enumerate(order)
            ranks[idx] = PreferenceIndex(pos)
        end
        return StrictRank{N}(SVector{N,PreferenceIndex}(Tuple(ranks)))
    else
        throw(ArgumentError("tie_break must be :error or a function (i,j,ranks,pool)->Bool"))
    end
end

function make_rank_bucket_linearizer(ranks::AbstractVector; rng = MersenneTwister(), missing_last::Bool = true)
    N = length(ranks)
    prio = randperm(rng, N)
    has = BitVector(undef, N)
    key = Vector{Int}(undef, N)
    @inbounds for i in 1:N
        v = ranks[i]
        if ismissing(v)
            has[i] = false; key[i] = 0
        else
            has[i] = true
            key[i] = v isa Integer ? Int(v) : Int(round(Integer, v))
        end
    end
    return (i, j, _, _) -> begin
        hi, hj = has[i], has[j]
        if hi != hj
            return missing_last ? hi : !hi
        elseif !hi
            return prio[i] < prio[j]
        else
            ki, kj = key[i], key[j]
            ki == kj ? (prio[i] < prio[j]) : (ki < kj)
        end
    end
end

function linearize_ranks(ranks::AbstractVector; rng = MersenneTwister(), missing_last::Bool = true)
    N = length(ranks)
    order = collect(1:N)
    f = make_rank_bucket_linearizer(ranks; rng, missing_last)
    sort!(order; lt = (i, j) -> f(i, j, nothing, nothing))
    out = Vector{Int}(undef, N)
    @inbounds for (pos, idx) in enumerate(order)
        out[idx] = pos
    end
    return out
end

"Permutation of candidate-ids from best to worst."
function to_perm(x::StrictRank{N}) where {N}
    inv = Vector{PreferenceIndex}(undef, N)
    @inbounds for i in 1:N
        inv[Int(x.ranks[i])] = PreferenceIndex(i)
    end
    return SVector{N,PreferenceIndex}(Tuple(inv))
end

function to_perm(x::WeakRank{N}) where {N}
    r = x.ranks
    order = sort(collect(1:N), by = i -> (ismissing(r[i]) ? (true, typemax(Int), i) : (false, Int(r[i]), i)))
    return SVector{N,PreferenceIndex}(Tuple(PreferenceIndex(i) for i in order))
end

"Convert ranks to explicit indifference classes (levels). Unranked last."
function to_weakorder(x::WeakRank{N}) where {N}
    groups = Dict{Int,Vector{CandidateId}}(); unranked = CandidateId[]
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
    return levels
end

function to_weakorder(x::StrictRank{N}) where {N}
    id_by_rank = Vector{CandidateId}(undef, N)
    perm = sortperm(1:N, by = i -> Int(x.ranks[i]))
    for (pos, cid) in enumerate(perm)
        id_by_rank[pos] = CandidateId(cid)
    end
    return [[id_by_rank[k]] for k in 1:N]
end

# ============================ Pairwise ===================================

@inline _ij_from_k(k, N) = ((k - 1) ÷ N + 1, (k - 1) % N + 1)

"Pairwise (maybe-missing) from ranks with an explicit extension policy."
function to_pairwise(x::WeakRank{N}, pool::CandidatePool{N}; policy::ExtensionPolicy=BottomPolicyMissing()) where {N}
    ext = policy
    E  = Union{Missing,Int8}
    data = ntuple(Val(N*N)) do k
        i, j = _ij_from_k(k, N)
        if i == j
            Int8(0)
        else
            compare_maybe(ext, x.ranks[i], x.ranks[j], i, j, x.ranks, pool)
        end
    end
    return Pairwise{N,E}(SMatrix{N,N,E}(data))
end

"Delegate StrictRank → WeakRank for pairwise construction."
to_pairwise(x::StrictRank{N}, pool::CandidatePool{N}; policy::ExtensionPolicy=BottomPolicyMissing()) where {N} =
    to_pairwise(to_weak(x), pool; policy)

# Back-compat adapter for symbol-based kwarg (can be removed later).
function to_pairwise(x::WeakRank{N}, pool::CandidatePool{N}; extension=:bottom) where {N}
    return to_pairwise(x, pool; policy = resolve_policy(extension))
end
to_pairwise(x::StrictRank{N}, pool::CandidatePool{N}; extension=:bottom) where {N} =
    to_pairwise(to_weak(x), pool; extension=extension)

# ============================ Predicates & utils ==========================

function rank(x::WeakRank{N}, pool::CandidatePool{N}, cand::Symbol) where {N}
    return x.ranks[pool[cand]]
end
function rank(x::StrictRank{N}, pool::CandidatePool{N}, cand::Symbol) where {N}
    return x.ranks[pool[cand]]
end

function prefers(x::WeakRank, pool::CandidatePool, a::Symbol, b::Symbol)
    ra = rank(x, pool, a); rb = rank(x, pool, b)
    (ismissing(ra) || ismissing(rb)) && return false
    return ra < rb
end
prefers(x::StrictRank, pool::CandidatePool, a::Symbol, b::Symbol) = rank(x, pool, a) < rank(x, pool, b)

function indifferent(x::WeakRank, pool::CandidatePool, a::Symbol, b::Symbol)
    ra = rank(x, pool, a); rb = rank(x, pool, b)
    (ismissing(ra) || ismissing(rb)) && return false
    return ra == rb
end
indifferent(::StrictRank, ::CandidatePool, ::Symbol, ::Symbol) = false

function ordered_candidates(x::StrictRank{N}, pool::CandidatePool{N}) where {N}
    perm = to_perm(x)
    return [pool.names[Int(i)] for i in perm]
end

function weakorder_symbol_groups(levels::Vector{Vector{CandidateId}}, pool::CandidatePool{N}) where {N}
    [map(i -> pool.names[Int(i)], lvl) for lvl in levels]
end

function tie_groups(x::WeakRank{N}, pool::CandidatePool{N}) where {N}
    levels = to_weakorder(x)
    return weakorder_symbol_groups(levels, pool)
end

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
function restrict(pp::Pairwise{N,T}, pool::CandidatePool{N}, subset) where {N,T}
    ids = _resolve_subset(pool, subset)
    new_pool, backmap = _restrict_pool(pool, ids)
    K = length(ids)
    data = ntuple(i -> ntuple(j -> pp.matrix[Int(backmap[i]), Int(backmap[j])], Val(K)), Val(K))
    return Pairwise{K,T}(SMatrix{K,K,T}(data)), new_pool, backmap
end

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

function restrict(x::StrictRank{N}, pool::CandidatePool{N}, subset) where {N}
    ids = _resolve_subset(pool, subset)
    new_pool, backmap = _restrict_pool(pool, ids)
    K = length(ids)
    # positions in original strict order
    inv = Vector{PreferenceIndex}(undef, N)
    @inbounds for i in 1:N
        inv[Int(x.ranks[i])] = PreferenceIndex(i)
    end
    keep = Set(Int.(backmap))
    filtered = [Int(inv[k]) for k in 1:N if Int(inv[k]) in keep]
    ranks = Vector{PreferenceIndex}(undef, K)
    @inbounds for (pos, cid) in enumerate(filtered)
        ranks[pos] = PreferenceIndex(pos) # pos is new rank; cid is original id
    end
    # remap position→candidate for the subset
    # (preserve relative order of the original strict ranking within subset)
    return StrictRank{K}(SVector{K,PreferenceIndex}(Tuple(ranks))), new_pool, backmap
end

# ---- subset helpers ----
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

function _restrict_pool(pool::CandidatePool{N}, ids::Vector{CandidateId}) where {N}
    K = length(ids)
    new_names = [pool.names[Int(i)] for i in ids]
    new_pool = CandidatePool(new_names)
    backmap = SVector{K,CandidateId}(Tuple(ids))
    return new_pool, backmap
end

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
