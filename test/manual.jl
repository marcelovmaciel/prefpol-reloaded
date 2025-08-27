
cd("..")


using Revise
using PrefPol

import PrefPol as pp

# ---------- helpers ----------
const CANDS = ["A","B","C","D"]
const DEMOS = ["Sex","Race"]
const CAND_SYMS = Symbol.(CANDS)


function toy_scores_df(; n=20)
    rng = pp.MersenneTwister(1234)
    # make some real scores with a few special codes 96-99 sprinkled in
    cols = Dict{String,Vector{Union{Missing,Float64}}}()
    for c in CANDS
        v = rand(rng, 1:10, n) .|> Float64

        # put a few special codes
        for code in (96.0, 97.0, 98.0, 99.0)


            v[rand(rng, 1:n, 1)] .= code

        end
        cols[c] = v
    end
    df = pp.DataFrame(cols)
    df.Sex = pp.categorical(rand(rng, ["F","M"], n))
    df.Race = pp.categorical(rand(rng, ["W","B","O"], n))
    df
end

df = toy_scores_df()



dist_df = pp.build_candidate_score_distributions(df, CANDS)


[pp.convert_keys_to_int(v) for v in  values(dist_df)]


# Minimal ranking helpers for later tests
function toy_ranking_dict()
    # A > B > C > D (10 best → rank 1)
    scores = Dict(:A=>10,:B=>8,:C=>7,:D=>5)
    return get_order_dict(scores)
end


pdf = pp.profile_dataframe(df; score_cols=CANDS, demo_cols=DEMOS, rng=pp.MersenneTwister(4))

pp.metadata!(pdf, "candidates", Symbol.(CANDS))

#pp.metadata(pdf)

pp.compress_rank_column!(pdf, Symbol.(CANDS); col=:profile)

pp.metadata(pdf)


pp.decode_profile_column!(pdf)



vm = Dict(:zero => [deepcopy(pdf)], :random => [deepcopy(pdf)], :mice => [deepcopy(pdf)])


pp.decode_each!(vm)

all(isa(vm[:zero][1].profile[1], Dict{Symbol,Int}) for _ in 1:1)




vm[:zero][1].profile[1]





# ========= testing polarization measures
