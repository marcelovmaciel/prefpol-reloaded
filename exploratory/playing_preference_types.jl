# =========================
# Interactive smoke tests for PrefPol (Minimal API)
# Paste & run line-by-line in the REPL
# =========================
using Revise
import PrefPol as pp

# ------------------------------------------------------------
# 1) CandidatePool & basic utilities
# ------------------------------------------------------------
p = pp.CandidatePool([:Alice, :Bob, :Carol, :Dave])

p

length(p)                # expect 4

p[:Alice]                # expect PreferenceIndex(1)
p[:Dave]                 # expect PreferenceIndex(4)

p[1]                     # expect :Alice
p[4]                     # expect :Dave

collect(keys(p))         # expect [:Alice, :Bob, :Carol, :Dave] (canonical order)
pp.candidates(p)         # exact order: [:Alice, :Bob, :Carol, :Dave]

pp.to_cmap(p)            # Dict(id=>name), e.g., Dict(0x0001=>:Alice, ...)

# ------------------------------------------------------------
# 2) WeakRank / StrictRank constructors & predicates
# ------------------------------------------------------------
# Strict complete (contiguous 1..N, no ties, no missing)
b_strict = pp.StrictRank(p, [1,2,3,4])




p  = pp.CandidatePool([:Alice,:Bob,:Carol,:Dave])

sr = pp.StrictRank(p,[:Bob, :Dave, :Alice, :Carol] )

pp.pretty(sr, p)   # REPL will display: StrictRank(Alice ≻ Bob ≻ Carol ≻ Dave)



# Truncated ballot (Dave unranked) — represented as WeakRank
b_trunc = pp.WeakRank(p, Dict(:Bob=>1, :Alice=>2))


wo = pp.to_weakorder(b_trunc)

pp.pretty(wo, p)                     # WeakOrder(Carol ≻ Alice ~ Bob ≻ Dave)

pp.pretty(wo, p; hide_unranked=true) # WeakOrder(Carol ≻ Alice ~ Bob)  (unranked omitted: Dave)



b_trunc

# Weak with a tie (Alice ~ Bob), Dave unranked
b_weak = pp.WeakRank(p, Dict(:Carol=>1, :Alice=>2, :Bob=>2))
b_weak

# Vector input as WeakRank
b_vec = pp.WeakRank(p, [1,2,3,4])
b_vec

# asdict (emits only present ranks)
pp.asdict(b_strict, p)        # full dict
pp.asdict(b_trunc, p)         # missing Dave

# rank / prefers / indifferent
pp.rank(b_strict, p, :Alice)              # 1
pp.prefers(b_strict, p, :Alice, :Bob)     # true
pp.indifferent(b_strict, p, :Alice, :Bob) # false

pp.rank(b_trunc, p, :Dave)                # missing
pp.prefers(b_trunc, p, :Dave, :Alice)     # false (missing)
pp.indifferent(b_weak, p, :Alice, :Bob)   # true

# Strictify weak ballots
try
    pp.to_strict(b_trunc; tie_break=:error)   # should throw (missing present)
catch e
    e
end

pp.to_strict(b_trunc; tie_break=:linearize)   # strict order, missing last
pp.to_strict(b_weak;  tie_break=:linearize)   # strict order, tie broken by id

# ------------------------------------------------------------
# 3) Permutations & conversions
# ------------------------------------------------------------
# Permutation of candidate ids (best→worst)
perm1 = pp.to_perm(b_strict)
perm1

perm2 = pp.to_perm(b_trunc)  # missing last
perm2

# Ordered candidate names from StrictRank
pp.ordered_candidates(b_strict, p)  # [:Alice,:Bob,:Carol,:Dave]

# ------------------------------------------------------------
# 4) WeakOrder representation & conversions
# ------------------------------------------------------------
wo = pp.to_weakorder(b_weak)
wo

wo.levels                           # e.g. [[3], [1,2], [4]] by ids
pp.weakorder_symbol_groups(wo, p)   # names at each level

# Linearize a WeakRank (equivalent of old `linearize`)
pp.to_strict(b_weak; tie_break=:linearize)

# ------------------------------------------------------------
# 5) Pairwise & extension policies
# ------------------------------------------------------------
# Default policy (:bottom): ranked ≻ unranked
pp1 = pp.to_pairwise(b_weak, p; extension=:bottom)
pp1.matrix

# Policy :none => any missing pair = 0
pp2 = pp.to_pairwise(b_trunc, p; extension=:none)
pp2.matrix

# Custom extension example: put unranked at top (unranked ≻ ranked)
_ext_top(ra, rb, i, j, ranks, pool) = (ismissing(ra) && ismissing(rb)) ? 0 :
                                      (ismissing(ra) ? 1 : (ismissing(rb) ? -1 :
                                      (Int(rb) - Int(ra) > 0 ? 1 : Int(rb) - Int(ra) < 0 ? -1 : 0)))
pp3 = pp.to_pairwise(b_trunc, p; extension=_ext_top)
pp3.matrix

# to_pairwise from different sources
pp.to_pairwise(b_strict, p).matrix
pp.to_pairwise(pp.WeakRank(p, Dict(:Alice=>1,:Bob=>2,:Carol=>3,:Dave=>4)), p).matrix
pp.to_pairwise(pp.to_strict(b_weak; tie_break=:linearize), p).matrix   # force strict first

# Reference to built-ins
pp.ext_bottom      # function
pp.ext_none        # function

# ------------------------------------------------------------
# 6) Restriction hooks
# ------------------------------------------------------------
subset_syms = [:Alice, :Carol, :Dave]
(new_pool, backmap) = pp.restrict(p, subset_syms)
new_pool
collect(backmap)   # new→old ids, expect [1,3,4]

# Restrict WeakRank
(new_b_trunc, np1, bm1) = pp.restrict(b_trunc, p, subset_syms)
new_b_trunc; np1; collect(bm1)
pp.asdict(new_b_trunc, np1)

# Restrict StrictRank (then show permutation)
(new_b_strict, np2, bm2) = pp.restrict(b_strict, p, subset_syms)
new_b_strict; np2; collect(bm2)
pp.to_perm(new_b_strict)

# Restrict WeakOrder
(new_wo, np3, bm3) = pp.restrict(wo, p, subset_syms)
new_wo.levels; np3; collect(bm3)

# Restrict Pairwise
ppair = pp.to_pairwise(b_weak, p; extension=:bottom)
(new_ppair, np4, bm4) = pp.restrict(ppair, p, subset_syms)
new_ppair.matrix; np4; collect(bm4)

# ------------------------------------------------------------
# 7) CSV readers (requires CSV.jl)
# ------------------------------------------------------------
# rank_columns CSV: columns are candidates, cells are ranks (or blank)
rank_csv = """
Alice,Bob,Carol,Dave
1,2,3,4
2,1,3,
,1,2,
"""
rank_path = mktemp()[1]; open(rank_path, "w") do io; write(io, rank_csv); end
ballots_rank, pool_rank = pp.read_rank_columns_csv(rank_path)
length(ballots_rank); pp.candidates(pool_rank)
ballots_rank[1]; pp.asdict(ballots_rank[2], pool_rank)

# candidate_columns CSV: columns are preference positions, cells are names
cand_csv = """
pref1,pref2,pref3
Alice,Bob,Carol
Bob,Alice,
Carol,,
"""
cand_path = mktemp()[1]; open(cand_path, "w") do io; write(io, cand_csv); end
ballots_cand, pool_cand = pp.read_candidate_columns_csv(cand_path; cols=[:pref1,:pref2,:pref3])
length(ballots_cand); pp.candidates(pool_cand)
ballots_cand[1]; pp.asdict(ballots_cand[2], pool_cand)

# ------------------------------------------------------------
# 8) Misc convenience checks
# ------------------------------------------------------------
np = new_pool
np[:Carol]   # expect PreferenceIndex(2) in the restricted pool
np[2]        # expect :Carol

# Show weak-order names for restricted data
pp.weakorder_symbol_groups(new_wo, np)

# Done. Evaluate any line again to re-inspect values.
