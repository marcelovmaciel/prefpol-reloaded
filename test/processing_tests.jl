using Test
using PrefPol
using DataFrames

import PrefPol: build_candidate_score_distributions, convert_keys_to_int,
    sanitize_countmaps, compute_dont_know_her,
    prepare_scores_for_imputation_int, prepare_scores_for_imputation_categorical,
    get_most_known_candidates, select_top_candidates, compute_candidate_set,
    get_df_just_top_candidates, GLOBAL_R_IMPUTATION, imputation_variants,
    weighted_bootstrap, get_row_candidate_score_pairs, get_order_dict,
    force_scores_become_linear_rankings, build_profile, profile_dataframe,
    dict2svec, decode_rank, perm2dict, perm_to_dict,
    compress_rank_column!, decode_profile_column!, decode_each!, load_spss_file


using CategoricalArrays
using Random
using StatsBase
using StaticArrays
using PooledArrays
using Impute
using RCall

@testset "preprocessing_general" begin
    @testset "load_spss_file" begin
        have_pkgs = try
            rcopy(Bool, RCall.reval("requireNamespace('haven', quietly=TRUE)")) &&
            rcopy(Bool, RCall.reval("requireNamespace('mice', quietly=TRUE)"))
        catch
            false
        end
        if have_pkgs
            path = tempname()*".sav"
            RCall.reval("library(haven); write_sav(data.frame(x=1:3), '$path')")
            df = load_spss_file(path)
            @test nrow(df) == 3
        else
            @test_throws Exception load_spss_file("missing.sav")
        end
    end

    @testset "build_candidate_score_distributions" begin
        df = DataFrame(A=[1,2,1], B=[3,3,3])
        cands = ["A","B"]
        cm = build_candidate_score_distributions(df, cands)
        @test cm["A"][1] == 2
        @test cm["B"][3] == 3
    end

    @testset "convert_keys_to_int" begin
        d = Dict(1=>2, 2.0=>3)
        cd = convert_keys_to_int(d)
        @test cd[2] == 3
        @test_throws ArgumentError convert_keys_to_int(Dict(2.5=>1))
        @test_throws ArgumentError convert_keys_to_int(Dict("1"=>1))
    end

    @testset "sanitize_countmaps" begin
        c = Dict("A"=>Dict(1.0=>2))
        s = sanitize_countmaps(c)
        @test s["A"][1] == 2
    end

    @testset "compute_dont_know_her" begin
        countmaps = Dict(
            "A"=>Dict(1=>3,96=>1),
            "B"=>Dict(2=>4,98=>2)
        )
        dkh = compute_dont_know_her(countmaps, 5)
        @test dkh == [("A",20.0),("B",40.0)]
    end

    @testset "prepare_scores_for_imputation_int" begin
        df = DataFrame(C1=[1.0,2.0,96.0], C2=[3.0,97.0,4.0])
        res = prepare_scores_for_imputation_int(df, ["C1","C2"])
        @test eltype(res.C1) <: Union{Missing,Int}
        @test res.C1[3] === missing
        @test res.C2[2] === missing
    end

    @testset "prepare_scores_for_imputation_categorical" begin
        df = DataFrame(C1=[1.0,2.0,96.0], C2=[3.0,97.0,4.0])
        res = prepare_scores_for_imputation_categorical(df, ["C1","C2"])
        println(res)
        @test CategoricalArrays.isordered(res.C1)
    end

    @testset "get_most_known_candidates" begin
        dkh = [("A",10.0),("B",20.0),("C",30.0)]
        @test get_most_known_candidates(dkh, 2) == ["A","B"]
    end

    @testset "select_top_candidates" begin
        countmaps = Dict(
            "A"=>Dict(1=>5,96=>2),
            "B"=>Dict(1=>6,96=>1),
            "C"=>Dict(1=>7,96=>0)
        )
        sel = select_top_candidates(countmaps,7; m=2, force_include=["A"])
        @test sel == ["A","C"]
    end

    @testset "compute_candidate_set" begin
        df = DataFrame(A=[1,96,2], B=[1,2,3], C=[1,96,96])
        set = compute_candidate_set(df; candidate_cols=["A","B","C"], m=2)
        @test set == ["B","A"]
    end

    @testset "get_df_just_top_candidates" begin
        df = DataFrame(A=1:3, B=4:6, C=7:9, Age=[20,30,40])
        PrefPol.dont_know_her = [("A",0.0),("B",10.0),("C",20.0)]
        df_top = get_df_just_top_candidates(df, 2; demographics=["Age"])
        @test names(df_top) == ["A","B","Age"]
        df_sel = get_df_just_top_candidates(df, ["A","C"]; demographics=["Age"])
        @test names(df_sel) == ["A","C","Age"]
    end

    @testset "GLOBAL_R_IMPUTATION" begin
        df = DataFrame(A=[1,2,97], B = [3,2,1])
        have_pkg = try
            rcopy(Bool, RCall.reval("requireNamespace('mice', quietly=TRUE)"))
        catch
            false
        end
        if have_pkg
            imp = GLOBAL_R_IMPUTATION(df)
            @test all(!ismissing, imp.A)
        else
            @test_throws Exception GLOBAL_R_IMPUTATION(df)
        end
    end

    @testset "imputation_variants" begin
        df = DataFrame(A=[1,96], B=[2,97], Age=[30,40])
        have_pkg = try
            rcopy(Bool, RCall.reval("requireNamespace('mice', quietly=TRUE)"))
        catch
            false
        end
        if have_pkg
            res = imputation_variants(df, ["A","B"], ["Age"])
            @test keys(res) == (:zero,:random,:mice)
        else
            @test_throws Exception imputation_variants(df, ["A","B"], ["Age"])
        end
    end

    @testset "weighted_bootstrap" begin
        df = DataFrame(A=1:3)
        w = [0.2,0.3,0.5]
        samples = weighted_bootstrap(df,w,2)
        @test length(samples) == 2
        @test all(nrow(s) == 3 for s in samples)
    end

    @testset "get_row_candidate_score_pairs" begin
        df = DataFrame(A=[1,2], B=[3,4])
        row = df[1,:]
        d = get_row_candidate_score_pairs(row, ["A","B"])
        @test d[:A] == 1
        @test d[:B] == 3
    end

    @testset "get_order_dict" begin
        d = Dict(:A=>10,:B=>5,:C=>5)
        o = get_order_dict(d)
        @test o[:A] == 1 && o[:B] == 2 && o[:C] == 2
    end

    @testset "force_scores_become_linear_rankings" begin
        d = Dict(:A=>3,:B=>3,:C=>1)
        rng = MersenneTwister(1)
        r = force_scores_become_linear_rankings(d; rng=rng)
        @test sort(collect(values(r))) == [1,2,3]
        @test keys(r) == keys(d)
    end

    @testset "build_profile" begin
        df = DataFrame(A=[1,2], B=[2,1])
        prof = build_profile(df; score_cols=["A","B"], rng=MersenneTwister(1), kind=:linear)
        @test length(prof) == 2
        @test all(haskey(p, :A) && haskey(p, :B) for p in prof)
    end

    @testset "profile_dataframe" begin
        df = DataFrame(A=[1,2], B=[2,1], Age=[30,40])
        pdf = profile_dataframe(df; score_cols=["A","B"], demo_cols=["Age"], rng=MersenneTwister(1))
        @test ((:profile in names(pdf)) || ("profile" in names(pdf)))
        @test (names(pdf) == [:profile, :Age]) ||  (names(pdf) == ["profile", "Age"])
    end

    @testset "dict2svec" begin
        d = Dict(:A=>1,:B=>2,:C=>3)
        cs = [:A,:B,:C]
        sv = dict2svec(d; cs=cs)
        @test sv == SVector{3,UInt8}(1,2,3)
    end

    @testset "decode_rank" begin
        pool = [SVector{3,UInt8}(1,2,3), SVector{3,UInt8}(1,3,2)]
        @test decode_rank(2,pool) == SVector{3,UInt8}(1,3,2)
        s = SVector{3,UInt8}(2,1,3)
        @test decode_rank(s, pool) === s
    end

    @testset "compress_rank_column!" begin
        df = DataFrame(profile=[Dict(:A=>1,:B=>2,:C=>3), Dict(:A=>2,:B=>1,:C=>3)])
        metadata!(df, "candidates", [:A,:B,:C])
        pool = compress_rank_column!(df, [:A,:B,:C])
        @test df.profile isa PooledArray
        @test pool isa Vector
    end

    @testset "perm2dict and perm_to_dict" begin
        perm = [2,1,3]
        cs = [:A,:B,:C]
        d = perm2dict(perm, cs)
        @test d == Dict(:A=>2,:B=>1,:C=>3)
        @test perm_to_dict(perm, cs) == d
    end

    @testset "decode_profile_column!" begin
        df = DataFrame(profile=[Dict(:A=>1,:B=>2,:C=>3)])
        metadata!(df, "candidates", [:A,:B,:C])
        pool = compress_rank_column!(df, [:A,:B,:C])
        decode_profile_column!(df)
        @test df.profile[1][:A] == 1
    end

    @testset "decode_each!" begin
        df = DataFrame(profile=[Dict(:A=>1,:B=>2,:C=>3)])
        metadata!(df, "candidates", [:A,:B,:C])
        pool = compress_rank_column!(df, [:A,:B,:C])
        vm = Dict("x"=>[df])
        decode_each!(vm)
        @test vm["x"][1].profile[1][:B] == 2
    end
end
