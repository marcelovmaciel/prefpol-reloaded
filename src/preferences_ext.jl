#### preferences_ext.jl — Extension policies (ranked vs. unranked; missing semantics)

# A policy must define:
#   (p::ExtensionPolicy)(ra, rb, i, j, ranks, pool)                :: Int8
#   compare_maybe(p::ExtensionPolicy, ra, rb, i, j, ranks, pool)   :: Union{Missing,Int8}
#
# Convention: lower rank value is better; `missing` = unranked.

abstract type ExtensionPolicy end

"Ranked beats unranked; both unranked tie. No missings returned."
struct BottomPolicy <: ExtensionPolicy end

"Strict-only comparisons; any missing inputs tie (0). No missings returned."
struct NonePolicy   <: ExtensionPolicy end

"Ranked beats unranked; both unranked → missing."
struct BottomPolicyMissing <: ExtensionPolicy end

"Strict-only; if any missing → missing."
struct NonePolicyMissing   <: ExtensionPolicy end

# -------- Total (no missing in outputs) --------
@inline function (::_p::BottomPolicy)(ra, rb, i, j, ranks, pool)::Int8
    if ismissing(ra) && ismissing(rb)
        return Int8(0)
    elseif ismissing(ra)
        return Int8(-1)
    elseif ismissing(rb)
        return Int8(1)
    else
        d = Int(rb) - Int(ra)
        return d > 0 ? Int8(1) : d < 0 ? Int8(-1) : Int8(0)
    end
end

@inline function (::_p::NonePolicy)(ra, rb, i, j, ranks, pool)::Int8
    if ismissing(ra) || ismissing(rb)
        return Int8(0)
    else
        d = Int(rb) - Int(ra)
        return d > 0 ? Int8(1) : d < 0 ? Int8(-1) : Int8(0)
    end
end

# -------- Maybe (allow missing in outputs) --------
@inline function compare_maybe(::BottomPolicyMissing, ra, rb, i, j, ranks, pool)::Union{Missing,Int8}
    if ismissing(ra) && ismissing(rb)
        return missing
    elseif ismissing(ra)
        return Int8(-1)
    elseif ismissing(rb)
        return Int8(1)
    else
        d = Int(rb) - Int(ra)
        return d > 0 ? Int8(1) : d < 0 ? Int8(-1) : Int8(0)
    end
end

@inline function compare_maybe(::NonePolicyMissing, ra, rb, i, j, ranks, pool)::Union{Missing,Int8}
    if ismissing(ra) || ismissing(rb)
        return missing
    else
        d = Int(rb) - Int(ra)
        return d > 0 ? Int8(1) : d < 0 ? Int8(-1) : Int8(0)
    end
end

# Back-compat resolver (optional): maps symbols to policies.
# Prefer passing policy instances at call sites.
resolve_policy(sym) = sym === :bottom      ? BottomPolicyMissing() :
                      sym === :none        ? NonePolicyMissing()   :
                      sym === :bottom_miss ? BottomPolicyMissing() :
                      sym === :none_miss   ? NonePolicyMissing()   :
                      throw(ArgumentError("Unknown extension policy: $sym"))
