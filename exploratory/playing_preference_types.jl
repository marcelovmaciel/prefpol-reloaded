# =========================
# Interactive smoke tests for PrefPol (paste & run line-by-line)
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
# 2) RankPreference constructors & predicates
# ------------------------------------------------------------
# Strict complete
b_strict = pp.RankPreference(p, Dict(:Alice=>1, :Bob=>2, :Carol=>3, :Dave=>4); style=pp.Strict)
b_strict

# Strict truncated (Dave unranked)
b_strict_trunc = pp.RankPreference(p, Dict(:Bob=>1, :Alice=>2, :Carol=>3); style=pp.Strict)
b_strict_trunc

# Weak with a tie (Alice ~ Bob), Dave unranked
b_weak = pp.RankPreference(p, Dict(:Carol=>1, :Alice=>2, :Bob=>2); style=pp.Weak)
b_weak

# Vector input, style=:auto
b_auto = pp.RankPreference(p, [1,2,3,4]; style=:auto)
typeof(b_auto)           # expect pp.RankPreference{pp.Strict,4,pp.PreferenceIndex}

# asdict (emits only present ranks)
pp.asdict(b_strict, p)        # expect full dict
pp.asdict(b_strict_trunc, p)  # expect missing Dave

# rank / prefers / indifferent
pp.rank(b_strict, p, :Alice)              # 1
pp.prefers(b_strict, p, :Alice, :Bob)     # true
pp.indifferent(b_strict, p, :Alice, :Bob) # false

pp.rank(b_strict_trunc, p, :Dave)         # missing
pp.prefers(b_strict_trunc, p, :Dave, :Alice) # false (missing)
pp.indifferent(b_weak, p, :Alice, :Bob)   # true

# ------------------------------------------------------------
# 3) OrderPreference & conversions
# ------------------------------------------------------------
# From names (strict permutation)
ord = pp.OrderPreference(p, [:Alice,:Bob,:Carol,:Dave])
ord

# From strict ranks
ord2 = pp.OrderPreference(b_strict, p)
ord2

# Functors
TRs = pp.ToRank{pp.Strict}(p)
TRw = pp.ToRank{pp.Weak}(p)
TOs = pp.ToOrder{pp.Strict}(p)
TOw = pp.ToOrder{pp.Weak}(p)

TRs(ord2)                       # -> RankPreference{Strict}
TOw(b_weak)                     # -> WeakOrder
pp.ordered_candidates(b_strict, p)  # [:Alice,:Bob,:Carol,:Dave]

# ------------------------------------------------------------
# 4) WeakOrder representation & conversions
# ------------------------------------------------------------
wo = pp.weakorder(b_weak)
wo

wo.levels                           # expect e.g. [[3], [1,2], [4]] by ids
pp.weakorder_symbol_groups(wo, p)   # names at each level
pp.indifference_list(b_weak)        # raw ID levels (same as wo.levels)

# WeakOrder -> ranks, and linearizations
pp.ranks_from_weakorder(wo)         # NTuple with equal ranks per level
pp.linearize(wo)                    # NTuple strict 1..N (by level then id)
pp.linearize(b_weak, p)             # RankPreference{Strict}

# ToRank from WeakOrder
TRw(wo)                             # RankPreference{Weak}
TRs(wo)                             # RankPreference{Strict}

# ------------------------------------------------------------
# 5) PairwisePreference & extension policies
# ------------------------------------------------------------
# Default policy (:bottom): ranked ≻ unranked
pp1 = pp.PairwisePreference(b_weak, p; extension=:bottom)
pp1.matrix

# Policy :none => any missing pair = 0
pp2 = pp.PairwisePreference(b_strict_trunc, p; extension=:none)
pp2.matrix

# Custom extension example: put unranked at top (unranked ≻ ranked)
_ext_top(ra, rb, i, j, ranks, pool) = (ismissing(ra) && ismissing(rb)) ? 0 :
                                      (ismissing(ra) ? 1 : (ismissing(rb) ? -1 :
                                      (Int(rb) - Int(ra) > 0 ? 1 : Int(rb) - Int(ra) < 0 ? -1 : 0)))
pp3 = pp.PairwisePreference(b_strict_trunc, p; extension=_ext_top)
pp3.matrix

# ToPairwise from different sources
TPs = pp.ToPairwise{pp.Strict}(p)
TPw = pp.ToPairwise{pp.Weak}(p)
TPs(b_strict).matrix
TPs(Dict(:Alice=>1,:Bob=>2,:Carol=>3,:Dave=>4)).matrix
TPs(ord).matrix
TPs(wo).matrix

# Reference to built-ins
pp.ext_bottom      # function
pp.ext_none        # function

# ------------------------------------------------------------
# 6) Restriction hooks
# ------------------------------------------------------------
# choose a subset/order (symbols)
subset_syms = [:Alice, :Carol, :Dave]
(new_pool, backmap) = pp.restrict(p, subset_syms)
new_pool
collect(backmap)   # new→old ids, expect [1,3,4]

# Restrict RankPreference (strict)
(new_b_strict, np1, bm1) = pp.restrict(b_strict, p, subset_syms)
new_b_strict; np1; collect(bm1)
pp.asdict(new_b_strict, np1)

# Restrict OrderPreference
(new_ord, np2, bm2) = pp.restrict(ord, p, subset_syms)
new_ord; np2; collect(bm2)

# Restrict WeakOrder
(new_wo, np3, bm3) = pp.restrict(wo, p, subset_syms)
new_wo.levels; np3; collect(bm3)

# Restrict PairwisePreference
ppair = pp.PairwisePreference(b_weak, p; extension=:bottom)
(new_ppair, np4, bm4) = pp.restrict(ppair, p, subset_syms)
new_ppair.matrix; np4; collect(bm4)

# ------------------------------------------------------------
# 7) CSV readers (optional; requires CSV.jl)
# ------------------------------------------------------------
# rank_columns CSV: columns are candidates, cells are ranks (or blank)
rank_csv = """
Alice,Bob,Carol,Dave
1,2,3,4
2,1,3,
,1,2,
"""
rank_path = mktemp()[1]; open(rank_path, "w") do io; write(io, rank_csv); end
ballots_rank, pool_rank = pp.read_rank_columns_csv(rank_path; style=:auto)
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
# name/id lookups again on restricted pool
np = new_pool
np[:Carol]   # expect PreferenceIndex(2) in the restricted pool
np[2]        # expect :Carol

# Show weak-order names for restricted data
pp.weakorder_symbol_groups(new_wo, np)

# Done. Evaluate any line again to re-inspect values.
