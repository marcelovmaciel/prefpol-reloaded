# =========================
# Interactive smoke tests
# =========================
using Revise
import PrefPol
const pp = PrefPol



# Small REPL helpers
pretty(x)    = pp.pretty(x, p)
strictify(x) = pp.to_strict(x; tie_break = pp.make_rank_bucket_linearizer(x.ranks))

# ------------------------------------------------------------
# 1) CandidatePool & basics
# ------------------------------------------------------------
p = pp.CandidatePool([:Alice, :Bob, :Carol, :Dave])

p
length(p)                # 4
p[:Alice]                # PreferenceIndex(1)
p[:Dave]                 # PreferenceIndex(4)
p[1]                     # :Alice
p[4]                     # :Dave
collect(keys(p))         # [:Alice, :Bob, :Carol, :Dave]
pp.candidates(p)         # same order
pp.to_cmap(p)            # Dict(id=>name)

# ------------------------------------------------------------
# 2) WeakRank / StrictRank constructors & predicates
# ------------------------------------------------------------
# Strict complete
b_strict = pp.StrictRank(p, [1,2,3,4])

# Strict from permutation of names
sr = pp.StrictRank(p, [:Bob, :Dave, :Alice, :Carol])
pp.pretty(sr, p)   # StrictRank(Bob ≻ Dave ≻ Alice ≻ Carol)

# Truncated ballot (Dave unranked) — WeakRank
b_trunc = pp.WeakRank(p, Dict(:Bob=>1, :Alice=>2))
pp.rank(b_trunc, p, :Alice)  # 2

# Weak-order view (levels as Vector{Vector{CandidateId}})
wo = pp.to_weakorder(b_trunc)
pretty(wo)                         # WeakOrder(Carol ≻ Alice ~ Bob ≻ Dave)
pp.pretty(wo, p; hide_unranked=true)  # hides last level if unranked exist

# Weak with a tie (Alice ~ Bob), Dave unranked
b_weak = pp.WeakRank(p, Dict(:Carol=>1, :Alice=>2, :Bob=>2))

# Vector input as WeakRank
b_vec = pp.WeakRank(p, [1,2,3,4])
pp.pretty(pp.to_weakorder(b_vec), p)

# asdict (present ranks only for WeakRank)
pp.asdict(b_strict, p)
pp.asdict(b_trunc,  p)

# rank / prefers / indifferent
pp.rank(b_strict, p, :Alice)                # 1
pp.prefers(b_strict, p, :Alice, :Bob)       # true
pp.indifferent(b_strict, p, :Alice, :Bob)   # false

pp.rank(b_trunc,  p, :Dave)                 # missing
pp.prefers(b_trunc, p, :Dave, :Alice)       # false (missing treated as not preferred)
pp.indifferent(b_weak, p, :Alice, :Bob)     # true

# Strictify weak ballots
try
    pp.to_strict(b_trunc; tie_break=:error)   # throws (missing present)
catch e
    e
end

println("WeakRank as WeakOrder (names):")
println(pretty(pp.to_weakorder(b_weak)))

# ------------------------------------------------------------
# 3) Pairwise checks (policies)
# ------------------------------------------------------------
# Preferred: pass explicit policy objects
pp_none = pp.to_pairwise(b_weak, p; policy = pp.NonePolicyMissing())     # any missing ⇒ missing
pp_bot  = pp.to_pairwise(b_weak, p; policy = pp.BottomPolicyMissing())   # ranked ≻ unranked; both unranked ⇒ missing

# Back-compat adapter also works (symbol → policy)
pp_none_bc = pp.to_pairwise(b_weak, p; extension = :none)
pp_bot_bc  = pp.to_pairwise(b_weak, p; extension = :bottom)

# Visual check (UI lives in preferences_display.jl)
pp.show_dodgson_table_color(pp_bot; pool = p)

pp_bot
pp_bot.matrix

# ------------------------------------------------------------
# 4) Permutations & conversions
# ------------------------------------------------------------
# Candidate-ids (best→worst)
perm1 = pp.to_perm(b_strict)
perm1

perm2 = pp.to_perm(b_trunc)  # unranked last
perm2

# Ordered candidate names from StrictRank
pp.ordered_candidates(b_strict, p)  # [:Alice,:Bob,:Carol,:Dave]

# ------------------------------------------------------------
# 5) Restriction hooks
# ------------------------------------------------------------
subset_syms = [:Alice, :Carol, :Dave]

# There is no standalone `restrict(::CandidatePool, ...)`.
# Use a preference restriction to obtain (new_pool, backmap).
(new_b_trunc, np1, bm1) = pp.restrict(b_trunc, p, subset_syms)
new_b_trunc; np1; collect(bm1)      # new→old ids
pp.asdict(new_b_trunc, np1)

# Restrict StrictRank
(new_b_strict, np2, bm2) = pp.restrict(b_strict, p, subset_syms)
new_b_strict; np2; collect(bm2)
pp.to_perm(new_b_strict)

# Restrict Pairwise
ppair = pp.to_pairwise(b_weak, p; policy = pp.BottomPolicyMissing())
(new_ppair, np3, bm3) = pp.restrict(ppair, p, subset_syms)
new_ppair.matrix; np3; collect(bm3)

# ------------------------------------------------------------
# 6) Weak-order name rendering on restricted pool
# ------------------------------------------------------------
pp.weakorder_symbol_groups(pp.to_weakorder(new_b_trunc), np1)

# ------------------------------------------------------------
# 7) Custom policy example
# ------------------------------------------------------------
# Unranked-at-top: missing ≻ ranked; both missing ⇒ missing
struct TopPolicy <: pp.ExtensionPolicy end
pp.compare_maybe(::TopPolicy, ra, rb, i, j, ranks, pool) =
    ismissing(ra) && ismissing(rb) ? missing :
    ismissing(ra) ? Int8(1) :
    ismissing(rb) ? Int8(-1) :
    (Int(rb) - Int(ra) > 0 ? Int8(1) : Int8(Int(rb) - Int(ra) < 0 ? -1 : 0))

pp_top = pp.to_pairwise(b_trunc, p; policy = TopPolicy())
pp_top.matrix

# Done — re-evaluate lines as needed.
