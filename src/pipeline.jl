
# ─────────────────────────────────────────────────────────────────────────────
# Scenario inside a TOML file
# ─────────────────────────────────────────────────────────────────────────────
"""
    Scenario

A named candidate subset used to **force-include** certain candidates when
constructing profiles. Scenarios come from the election TOML and are used in
`generate_profiles_for_year*` to pin the candidate set logic for each plot/run.

# Fields
- `name::String`: Scenario identifier (e.g., "ideology", "front_runners").
- `candidates::Vector{String}`: Ordered list of candidate *ids/names* that must
  be present (and ordered first when trimming to `m` alternatives).
"""
struct Scenario
    name       :: String
    candidates :: Vector{String}
end

# ─────────────────────────────────────────────────────────────────────────────
# FULL election specification (= everything in the TOML)
# ─────────────────────────────────────────────────────────────────────────────
"""
    ElectionConfig

Full configuration parsed from the TOML for a given election year.

This struct travels alongside bootstrap data throughout the pipeline and encodes
how to load raw data, which candidates/demographics to keep, bootstrap settings,
and the list of `Scenario`s to generate.

# Fields
- `year::Int`: Election year.
- `data_loader::String`: Name of a loader function in the current module.
- `data_file::String`: Absolute path to the raw data file.
- `max_candidates::Int`: Upper bound for candidate set discovery.
- `m_values_range::Vector{Int}`: Values of `m` (number of alternatives) to plot.
- `n_bootstrap::Int`: Number of bootstrap replicates (default; can be overridden).
- `n_alternatives::Int`: Number of alternatives expected by some loaders.
- `rng_seed::Int`: Seed for reproducibility.
- `candidates::Vector{String}`: All candidate columns to consider.
- `demographics::Vector{String}`: Demographic columns used for grouping.
- `scenarios::Vector{Scenario}`: List of forced scenarios from the TOML.
"""
struct ElectionConfig
    year            :: Int
    data_loader     :: String
    data_file       :: String
    max_candidates  :: Int
    m_values_range  :: Vector{Int}

    n_bootstrap     :: Int          # default; can be overridden later
    n_alternatives  :: Int
    rng_seed        :: Int

    candidates      :: Vector{String}
    demographics    :: Vector{String}
    scenarios       :: Vector{Scenario}    # list of Scenario structs
end

"""
    load_election_cfg(path) -> ElectionConfig

Parse a TOML file describing an election and return an `ElectionConfig`.

- Resolves `data_file` relative to the **project root** unless already absolute.
- Converts `forced_scenarios` into a vector of `Scenario` objects.

This is the canonical entry point for configuration used by bootstrap and
profile generation.
"""
function load_election_cfg(path::AbstractString)::ElectionConfig
    t = TOML.parsefile(path)
    proj = dirname(Pkg.project().path)       # project root
    rawfile = isabspath(t["data_file"]) ? t["data_file"] :
              joinpath(proj, t["data_file"])

    scen_vec = [Scenario(s["name"], Vector{String}(s["candidates"]))
                for s in t["forced_scenarios"]]

    ElectionConfig(
        t["year"], t["data_loader"], rawfile,
        t["max_candidates"], Vector{Int}(t["m_values_range"]),
        t["n_bootstrap"], t["n_alternatives"],
        t["rng_seed"],
        Vector{String}(t["candidates"]),
        Vector{String}(t["demographics"]),
        scen_vec,
    )
end

import Base: show

const IND = "    "                       # 4-space indent

# ────────────────────────── helpers ──────────────────────────
"""
    _pp(io, key, val, lvl)

Pretty-print a single `key = value` line with indentation level `lvl`.
Internal helper used by the custom `show` methods below.
"""
_pp(io, key, val, lvl) = println(io, repeat(IND, lvl), key, " = ", val)

"""
    _pp_vec(io, key, vec, lvl; max=8)

Pretty-print up to the first `max` elements of `vec` as `key = [...]`.
Appends an elision marker with total length when the vector is longer.
"""
function _pp_vec(io, key, vec, lvl; max=8)
    head = first(vec, max)
    tail = length(vec) > max ? " … ("*string(length(vec))*")" : ""
    println(io, repeat(IND, lvl), key, " = ", head, tail)
end

# ────────────────────────── Scenario ─────────────────────────
"""
    show(io, ::MIME"text/plain", s::Scenario)

Pretty printer for `Scenario`, shown as `Scenario("name", [c1, c2, …])`.
"""
function show(io::IO, ::MIME"text/plain", s::Scenario; kwargs...)
    print(io, "Scenario(\"", s.name, "\", ", s.candidates, ")")
end

# ─────────────────────── ElectionConfig ──────────────────────
"""
    show(io, ::MIME"text/plain", ec::ElectionConfig)

Multi-line pretty printer for `ElectionConfig`, including a compact preview of
candidate and scenario lists.
"""
function show(io::IO, ::MIME"text/plain", ec::ElectionConfig; kwargs...)
    println(io, "ElectionConfig(")
    _pp(io, "year",             ec.year, 1)
    _pp(io, "data_loader",      ec.data_loader, 1)
    _pp(io, "data_file",        ec.data_file, 1)
    _pp(io, "max_candidates",   ec.max_candidates, 1)
    _pp(io, "m_values_range",   ec.m_values_range, 1)
    _pp(io, "n_bootstrap(def)", ec.n_bootstrap, 1)
    _pp(io, "n_alternatives",   ec.n_alternatives, 1)
    _pp(io, "rng_seed(def)",    ec.rng_seed, 1)
    _pp_vec(io, "candidates",   ec.candidates, 1)
    _pp(io, "demographics",     ec.demographics, 1)
    println(io, IND, "scenarios = [")
    for sc in ec.scenarios
        println(io, repeat(IND,2), sc)           # uses Scenario show
    end
    println(io, IND, "]")
    print(io, ")")
end

"""
    load_election_data(cfg) -> DataFrame

Resolve and call the configured `data_loader` in the **current module**,
returning a `DataFrame` with candidate and demographic columns. The loader is
looked up dynamically by name (`cfg.data_loader`) and called as

```julia
loader_fun(cfg.data_file; candidates = cfg.candidates)
```
"""
function load_election_data(cfg::ElectionConfig)
    loader_sym = Symbol(cfg.data_loader)

    # resolve in the current module’s namespace
    if !isdefined(@__MODULE__, loader_sym)
        throw(ArgumentError("data_loader ‘$(cfg.data_loader)’ not found in module $(nameof(@__MODULE__))"))
    end

    loader_fun = getfield(@__MODULE__, loader_sym)
    return loader_fun(cfg.data_file; candidates = cfg.candidates)
end

"""
    weighted_bootstrap(cfg::ElectionConfig) -> Dict{Symbol,Vector{DataFrame}}

Load raw election data via `load_election_data(cfg)`, slice to the configured
candidate and demographic columns, and run weighted bootstrapping using the
`peso` column as replicate weights. Returns a dictionary of bootstrap
replicates per imputation variant **only if** the underlying `weighted_bootstrap`
method dispatches on `(df, weights, B)`; here we simply delegate.
"""
function weighted_bootstrap(ecfg::ElectionConfig)
    df = load_election_data(ecfg)
    weights = df.peso
    B = ecfg.n_bootstrap
    # slice df to only candidates and demographics from ecfg
    candidates = ecfg.candidates
    demographics = ecfg.demographics
    df = select(df, candidates..., demographics...)
    bts = weighted_bootstrap(df, weights, B)
    return bts
end

const INT_DIR = "intermediate_data"
mkpath(INT_DIR)


"""
    save_bootstrap(cfg; dir = INT_DIR, overwrite = false, quiet = false)
        → NamedTuple{(:path,:data,:cached), ...}

Ensure a weighted-bootstrap exists for `cfg`:

  • If `dir/boot_YEAR.jld2` is missing *or* `overwrite=true`, build the
    bootstrap with `weighted_bootstrap(cfg)` and write it to disk.

  • Otherwise reuse the cached file, **loading the replicates** so that
    `.data` is never `nothing`.

Returned fields
---------------
| field   | meaning                                   |
|---------|-------------------------------------------|
| `path`  | full path to the `.jld2` file             |
| `data`  | the `reps` object (always in memory)      |
| `cached`| `true` if we reused an existing file      |
"""
function save_bootstrap(cfg::ElectionConfig;
                        dir::AbstractString = INT_DIR,
                        overwrite::Bool = false,
                        quiet::Bool = false)

    path = joinpath(dir, "boot_$(cfg.year).jld2")

    # ------------------ cache hit ------------------
    if !overwrite && isfile(path)
        !quiet && @warn "Reusing cached bootstrap at $(path); loading into memory"
        reps = nothing
        @load path reps                           # brings `reps` back
        return (path = path, data = reps, cached = true)
    end

    # ------------------ (re)build ------------------
    reps = weighted_bootstrap(cfg)                # heavy call
    @save path reps cfg
    !quiet && @info "Saved bootstrap for year $(cfg.year) → $(path)"
    return (path = path, data = reps, cached = false)
end



"""
    save_all_bootstraps(; years = nothing,
                            cfgdir = "config",
                            overwrite = false) -> Dict{Int,String}

Iterate over every `*.toml` in `cfgdir`; for each year that matches
`years` (or *all* years if `years === nothing`) build & save the bootstrap.

Returns a dictionary `year ⇒ saved_filepath`.
"""
function save_all_bootstraps(; years = nothing,
                             cfgdir::AbstractString = "config",
                             overwrite::Bool = false)
    # discover configs on disk
    toml_files = filter(p -> endswith(p, ".toml"), readdir(cfgdir; join=true))
    isempty(toml_files) && error("No TOML files found in $(cfgdir)")

    wanted = years === nothing        ? nothing           :
             isa(years, Integer)      ? Set([years])      :
             Set(years)

    saved = Dict{Int,String}()

    for f in sort(toml_files)
        cfg = load_election_cfg(f)
        (wanted !== nothing && !(cfg.year in wanted)) && continue
        @info "Processing year $(cfg.year) from $(f)"
        saved[cfg.year] = save_bootstrap(cfg; overwrite).path
    end
    return saved
end

"""
    load_all_bootstraps(; years = nothing,
                           dir   = INT_DIR,
                           quiet = false)
        → OrderedDict{Int,NamedTuple}

Read every `boot_YYYY.jld2` in `dir` (or just the chosen `years`)
and return them in a year-sorted `OrderedDict`.

Each value is a `NamedTuple` with

| field   | meaning                     |
|---------|-----------------------------|
| `data`  | the bootstrap replicates    |
| `cfg`   | the `ElectionConfig` object |
| `path`  | full path to the file       |
"""
function load_all_bootstraps(; years   = nothing,
                             dir::AbstractString = INT_DIR,
                             quiet::Bool = false)

    paths = filter(p -> occursin(r"boot_\d+\.jld2$", p),
                   readdir(dir; join = true))

    isempty(paths) && error("No bootstrap files found in $(dir)")

    selected = years === nothing       ? nothing :
               isa(years,Integer)      ? Set([years]) :
               Set(years)

    out = OrderedCollections.OrderedDict{Int,NamedTuple}()

    for f in sort(paths)                       # alphabetical = chronological
        yr = parse(Int, match(r"boot_(\d{4})\.jld2", basename(f)).captures[1])
        (selected !== nothing && !(yr in selected)) && continue

        reps = cfg = nothing
        @load f reps cfg

        !quiet && @info "Loaded bootstrap $(yr)  ←  $(f)"

        out[yr] = (data = reps, cfg = cfg, path = f)
    end
    return out
end

"""
    ImputedYear

Small index object that records, for one `year`, where each imputed replicate
is stored on disk for each variant.

# Fields
- `year::Int`
- `paths::Dict{Symbol,Vector{String}}`: e.g., `:mice => ["imp_2022_rep1_mice.jld2", …]`
"""
struct ImputedYear
    year::Int
    # Dict(:zero => [path1, path2, …], :random => …, :mice => …)
    paths::Dict{Symbol,Vector{String}}
end

"""
    getrep(iy::ImputedYear, variant::Symbol, i::Int) -> DataFrame

Load the *i*-th replicate of `variant` for that year.
"""
function getrep(iy::ImputedYear, variant::Symbol, i::Int)
    p = iy.paths[variant][i]
    df = nothing; @load p df        # `df` is how we store it below
    return df
end

Base.getindex(iy::ImputedYear, variant::Symbol, i::Int) = getrep(iy, variant, i)

const IMP_DATA_DIR = joinpath(INT_DIR, "imputed_data"); mkpath(IMP_DATA_DIR)

"""
    impute_bootstrap_to_files(path_boot;
                              imp_dir=IMP_DATA_DIR,
                              overwrite=false,
                              most_known_candidates=String[]) -> String

Run all imputation variants for every bootstrap replicate stored in
`path_boot::String`, save each imputed DataFrame to JLD2, and write a compact
`ImputedYear` index (`index_YEAR.jld2`). Returns the index file path.
"""
function impute_bootstrap_to_files(path_boot::String;
                                   imp_dir::AbstractString = IMP_DATA_DIR,
                                   overwrite::Bool         = false,
                                   most_known_candidates   = String[])

    reps = cfg = nothing
    @load path_boot reps cfg                 # same vars saved by save_bootstrap
    year = cfg.year

    # ---------------- per-variant path collectors -----------------
    var_syms = (:zero, :random, :mice)
    paths_dict = Dict(var => Vector{String}(undef, length(reps)) for var in var_syms)

    for (i, df_raw) in enumerate(reps)
        imp = imputation_variants(df_raw, cfg.candidates, cfg.demographics;
                                  most_known_candidates)

        for var in var_syms
            file = joinpath(imp_dir,
                    "imp_$(year)_rep$(i)_$(String(var)).jld2")
            if !overwrite && isfile(file)
                @warn "reusing $(file)"
            else
                df = imp[var]                                # DataFrame
                @save file df
            end
            paths_dict[var][i] = file
        end

        # ---------- free memory for this replicate ----------
        imp = reps[i] = df_raw = nothing
        GC.gc()
    end

    # tiny index object
    index = ImputedYear(year, paths_dict)
    ind_file = joinpath(imp_dir, "index_$(year).jld2")
    @save ind_file index

    @info "Finished imputation for $(year) → $(ind_file)"
    return ind_file
end

"""
    impute_all_bootstraps(; years=nothing, base_dir=INT_DIR, imp_dir=IMP_DATA_DIR,
                           overwrite=false, most_known_candidates=String[])
        -> OrderedDict{Int,String}

Batch version of `impute_bootstrap_to_files` over all `boot_YYYY.jld2` found in
`base_dir`, optionally filtered by `years`. Returns `year => index_path`.
"""
function impute_all_bootstraps(; years = nothing,
                               base_dir = INT_DIR,
                               imp_dir  = IMP_DATA_DIR,
                               overwrite = false,
                               most_known_candidates = String[])

    rx = r"boot_(\d{4})\.jld2$"
    files = filter(p -> occursin(rx, basename(p)), readdir(base_dir; join=true))
    isempty(files) && error("No bootstrap files found in $(base_dir)")

    wanted = years === nothing ? nothing :
             isa(years,Integer) ? Set([years]) : Set(years)

    worklist = Tuple{Int,String}[]
    for p in files
        yr = parse(Int, match(rx, basename(p)).captures[1])
        (wanted !== nothing && yr ∉ wanted) && continue
        push!(worklist, (yr, p))
    end
    sort!(worklist; by = first)

    prog = pm.Progress(length(worklist); desc = "Imputing bootstraps", barlen = 30)
    out  = OrderedDict{Int,String}()

    for (yr, p) in worklist
        @info "Imputing year $yr …"
        out[yr] = impute_bootstrap_to_files(p;
                     imp_dir = imp_dir,
                     overwrite = overwrite,
                     most_known_candidates = most_known_candidates)
        GC.gc()
        pm.next!(prog)
    end
    return out
end

"""
    _impute_year_to_files(reps, cfg; imp_dir=IMP_DATA_DIR, overwrite=false,
                          most_known_candidates=String[]) -> String

Internal worker that mirrors `impute_bootstrap_to_files`, but receives
in‑memory bootstrap `reps::Vector{DataFrame}` and an `ElectionConfig` instead
of a `boot_YYYY.jld2` file. Returns the written index path.
"""
function _impute_year_to_files(reps::Vector{DataFrame},
                               cfg::ElectionConfig;
                               imp_dir::AbstractString = IMP_DATA_DIR,
                               overwrite::Bool = false,
                               most_known_candidates = String[])

    year      = cfg.year
    idxfile   = joinpath(imp_dir, "index_$(year).jld2")
    variants  = (:zero, :random, :mice)
    nboot     = length(reps)

    # ────────────────────────────────────────────────────────────────────
    # 1. Fast path: cached index exists
    # ────────────────────────────────────────────────────────────────────
    if !overwrite && isfile(idxfile)
        @info "Reusing cached imputed index for year $year → $(idxfile)"
        return idxfile
    end

    # helper that lists all expected replicate files
    expected_path(var, i) = joinpath(
        imp_dir, "imp_$(year)_rep$(i)_$(String(var)).jld2")

    # ────────────────────────────────────────────────────────────────────
    # 2.  Could we build the index without recomputation?
    # ────────────────────────────────────────────────────────────────────
    if !overwrite
        all_exist = true
        for i in 1:nboot, var in variants
            isfile(expected_path(var, i)) || (all_exist = false; break)
        end
        if all_exist
            paths = Dict(var => [expected_path(var, i) for i in 1:nboot]
                         for var in variants)
            index = ImputedYear(year, paths)
            @save idxfile index
            @info "Rebuilt index for year $year without re-imputation."
            return idxfile
        end
    end

    # ────────────────────────────────────────────────────────────────────
    # 3.  Run full imputation (some files missing or overwrite=true)
    # ────────────────────────────────────────────────────────────────────
    @info "Running imputation for year $year …"
    paths = Dict(var => Vector{String}(undef, nboot) for var in variants)

    for i in 1:nboot
        df_raw = reps[i]
        imp    = imputation_variants(df_raw,
                                     cfg.candidates,
                                     cfg.demographics;
                                     most_known_candidates)

        for var in variants
            file = expected_path(var, i)
            if overwrite || !isfile(file)
                df = imp[var]
                @save file df
            end
            paths[var][i] = file
        end

        imp    = df_raw = nothing        # local cleanup
        GC.gc()
    end

    index = ImputedYear(year, paths)
    @save idxfile index
    @info "Saved imputed index for year $year → $(idxfile)"
    return idxfile
end

# ---------------------------------------------------------------------
# 2 · top-level driver starting from your in-memory `f3`
# ---------------------------------------------------------------------
"""
    impute_from_f3(f3; years=nothing, imp_dir=IMP_DATA_DIR, overwrite=false,
                   most_known_candidates=String[]) -> OrderedDict{Int,String}

Top‑level driver when you already have the `f3` object in memory. For each
requested `year`, runs `_impute_year_to_files` and returns `year => index_path`.
"""
function impute_from_f3(f3::OrderedDict;
                        years = nothing,
                        imp_dir::AbstractString = IMP_DATA_DIR,
                        overwrite::Bool = false,
                        most_known_candidates = String[])

    wanted = years === nothing        ? sort(collect(keys(f3))) :
             isa(years,Integer)       ? [years]                 :
             sort(collect(years))

    prog = pm.Progress(length(wanted); desc = "Imputing bootstraps", barlen = 30)
    out  = OrderedDict{Int,String}()

    for yr in wanted
        entry = f3[yr]                       # (data = reps, cfg = cfg, path = …)
        reps  = entry.data
        cfg   = entry.cfg

        @info "Imputing year $yr …"
        out[yr] = _impute_year_to_files(reps, cfg;
                                        imp_dir    = imp_dir,
                                        overwrite  = overwrite,
                                        most_known_candidates = most_known_candidates)

        GC.gc()                             # reclaim before next year
        pm.next!(prog)
    end
    return out
end

const IMP_PREFIX  = "boot_imp_"   # change here if you rename files
const IMP_DIR     = INT_DIR       # default directory to look in

# ————————————————————————————————————————————————————————————————
# 1.  Single-year loader
# ————————————————————————————————————————————————————————————————
"""
    load_imputed_bootstrap(year;
                           dir   = IMP_DIR,
                           quiet = false)  -> NamedTuple

Load `dir/boot_imp_YEAR.jld2` and return the stored NamedTuple
`(data = Dict, cfg = ElectionConfig, path = String)`.
"""
function load_imputed_bootstrap(year::Integer;
                                dir::AbstractString = IMP_DIR,
                                quiet::Bool = false)

    path = joinpath(dir, "$(IMP_PREFIX)$(year).jld2")
    isfile(path) || error("File not found: $(path)")

    imp = nothing
    @load path imp
    !quiet && @info "Loaded imputed bootstrap for year $(year) ← $(path)"
    return imp
end

"""
    load_imputed_year(year; dir=IMP_DATA_DIR) -> ImputedYear

Load the `index_YEAR.jld2` produced by imputation and return the `ImputedYear`
object for convenient access to per‑variant replicate paths.
"""
function load_imputed_year(year::Int;
                           dir::AbstractString = IMP_DATA_DIR)::ImputedYear
    idxfile = joinpath(dir, "index_$(year).jld2")
    isfile(idxfile) || error("index file not found: $(idxfile)")
    return JLD2.load(idxfile, "index")    # returns ImputedYear struct
end

# TODO: later, write a variant that takes just imp and config
# and loads f3 from disk, cakculate the sets, cleans it from disk, and proceeds

const CANDLOG = joinpath(INT_DIR, "candidate_set_warnings.log")

"""
    generate_profiles_for_year(year, f3_entry, imps_entry) -> OrderedDict

For each `Scenario` and each `m ∈ cfg.m_values_range`, build a profile
`DataFrame` for every imputation variant and replicate:
- compute candidate set consistent with the scenario and bootstrap draw,
- build `profile_dataframe`, compress and attach `metadata!(…, "candidates", …)`.

Returns a nested `OrderedDict`:
`scenario ⇒ m ⇒ (variant ⇒ Vector{DataFrame})`.
"""
function generate_profiles_for_year(year::Int,
                                    f3_entry::NamedTuple,
                                    imps_entry::NamedTuple)

    cfg            = f3_entry.cfg
    reps_raw       = f3_entry.data
    variants_dict  = imps_entry.data
    m_values       = cfg.m_values_range

    result = OrderedDict{String,OrderedDict{Int,OrderedDict{Symbol,Vector{DataFrame}}}}()

    for scen in cfg.scenarios
        sets = unique(map(df ->
            compute_candidate_set(df;
                candidate_cols = cfg.candidates,
                m              = cfg.max_candidates,
                force_include  = scen.candidates),
            reps_raw))

        length(sets) != 1 && @warn "Year $year, scenario $(scen.name): $(length(sets)) candidate sets; using first."
        full_list = sets[1]

        m_map = OrderedDict{Int,OrderedDict{Symbol,Vector{DataFrame}}}()

        for m in m_values
            trimmed  = Symbol.(first(full_list, m))          # ordered Vector{String}
            var_map  = OrderedDict{Symbol,Vector{DataFrame}}()

            for (variant, reps_imp) in variants_dict
                profiles = Vector{DataFrame}(undef, length(reps_imp))

                for (i, df_imp) in enumerate(reps_imp)
                    df = profile_dataframe(
                             df_imp;
                             score_cols = trimmed,
                             demo_cols  = cfg.demographics)
                    compress_rank_column!(df, trimmed; col = :profile)
                    # ------------- NEW LINE ----------------------------------
                    metadata!(df, "candidates", Symbol.(trimmed))
                    # ---------------------------------------------------------

                    profiles[i] = df
                end
                var_map[variant] = profiles
            end
            m_map[m] = var_map
        end
        result[scen.name] = m_map
    end
    return result
end

# ─────────────────────────────────────────────────────────────────────────────
# directories / tiny types (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
const PROFILES_DATA_DIR = joinpath(INT_DIR, "profiles_data")
mkpath(PROFILES_DATA_DIR)

"""
    ProfilesSlice

Lightweight handle to per‑replicate profile files for a fixed `(year, scenario, m)`.

# Fields
- `year::Int`
- `scenario::String`
- `m::Int`
- `cand_list::Vector{Symbol}`: ordered candidate list used for (en/de)coding
- `paths::Dict{Symbol,Vector{String}}`: `variant ⇒ replicate paths`
"""
struct ProfilesSlice
    year::Int
    scenario::String
    m::Int
    cand_list::Vector{Symbol}              # ordered for (en/de)coding
    paths::Dict{Symbol,Vector{String}}     # variant ⇒ file paths
end

Base.getindex(ps::ProfilesSlice, var::Symbol, i::Int) = begin
    p = ps.paths[var][i]; df = nothing; JLD2.@load p df; df
end

"""
    generate_profiles_for_year_streamed_from_index(year, f3_entry, iy;
                                                   out_dir=PROFILES_DATA_DIR,
                                                   overwrite=false)

Streaming variant of `generate_profiles_for_year` that reads imputed replicates
on demand from an `ImputedYear` index, writing each profile `DataFrame` to disk
and returning an index:
`scenario ⇒ m ⇒ ProfilesSlice`.
"""
function generate_profiles_for_year_streamed_from_index(
            year::Int,
            f3_entry::NamedTuple,
            iy::ImputedYear;
            out_dir::AbstractString = PROFILES_DATA_DIR,
            overwrite::Bool         = false)

    cfg            = f3_entry.cfg
    reps_raw       = f3_entry.data
    m_values       = cfg.m_values_range
    variants       = collect(keys(iy.paths))           # e.g. (:zero,:random,:mice)
    n_by_var       = Dict(v => length(iy.paths[v]) for v in variants)

    result = OrderedDict{String,OrderedDict{Int,ProfilesSlice}}()

    for scen in cfg.scenarios
        sets = unique(map(df ->
            compute_candidate_set(df;
                candidate_cols = cfg.candidates,
                m              = cfg.max_candidates,
                force_include  = scen.candidates),
            reps_raw))
        length(sets) != 1 && @warn "Year $year, scenario $(scen.name): " *
                                   "$(length(sets)) candidate sets; using the first."
        full_cset = sets[1]

        scen_map = OrderedDict{Int,ProfilesSlice}()

        for m in m_values
            cand_syms  = Symbol.(first(full_cset, m))
            paths_prof = Dict(v => Vector{String}(undef, n_by_var[v]) for v in variants)

            rep_counter = 0      # throttle GC

            for var in variants
                n_rep = n_by_var[var]

                for i in 1:n_rep
                    fprof = joinpath(out_dir,
                             "prof_$(year)_$(scen.name)_m$(m)_rep$(i)_" *
                             "$(String(var)).jld2")

                    # -------- fast‑skip if file already present --------------
                    if !overwrite && isfile(fprof)
                        paths_prof[var][i] = fprof
                        @debug "exists, skipping $(basename(fprof))"
                        continue
                    end

                    # -------- otherwise build & save -------------------------
                    df_imp = iy[var, i]

                    df = profile_dataframe(df_imp;
                            score_cols = cand_syms,
                            demo_cols  = cfg.demographics)
                    compress_rank_column!(df, cand_syms; col = :profile)
                    metadata!(df, "candidates", cand_syms)

                    JLD2.@save fprof df
                    @info "writing $(basename(fprof))"
                    paths_prof[var][i] = fprof

                    df = df_imp = nothing
                    rep_counter += 1
                    rep_counter % 10 == 0 && GC.gc()
                end
            end

            slice = ProfilesSlice(year, scen.name, m, cand_syms, paths_prof)
            scen_map[m] = slice
        end
        result[scen.name] = scen_map
    end

    idxfile = joinpath(out_dir, "profiles_index_$(year).jld2")
    JLD2.@save idxfile result
    @info "Encoded profiles for $year written; index at $(idxfile)"
    return result
end

const PROFILE_FILE = joinpath(INT_DIR, "all_profiles.jld2")

const PROFILE_DIR = joinpath(INT_DIR, "profiles")
mkpath(PROFILE_DIR)   # ensure it exists

# ────────────────────────────────────────────────────────────────────────────────
"""
    save_or_load_profiles_for_year(year, f3, imps;
                                  dir       = PROFILE_DIR,
                                  overwrite = false,
                                  verbose   = true)

For the given `year`, if `dir/profiles_YEAR.jld2` exists and `overwrite=false`,
issues a warning and loads it.  Otherwise:

  • Calls `generate_profiles_for_year(year, f3[year], imps[year])`
  • Saves the result as `profiles` in `dir/profiles_YEAR.jld2`
  • Returns the `profiles` object.
"""
function save_or_load_profiles_for_year(year::Int,
                                        f3,
                                        imps;
                                        dir::AbstractString = PROFILE_DIR,
                                        overwrite::Bool     = false,
                                        verbose::Bool       = true)

    path = joinpath(dir, "profiles_$(year).jld2")

    if isfile(path) && !overwrite
        verbose && @warn "Profiles for $year already exist at $path; loading cache."
        profiles = nothing
        @load path profiles
        return profiles
    end

    verbose && @info "Generating profiles for year $year…"
    profiles = generate_profiles_for_year(year, f3[year], imps[year])
    @save path profiles
    verbose && @info "Saved profiles for year $year → $path"
    return profiles
end

"Make a Dict<measure,<variant,Vector>> skeleton with empty vectors."
function init_accumulator(var_syms, measure_syms)
    accum = Dict{Symbol,Dict{Symbol,Vector{Float64}}}()
    for meas in measure_syms
        inner = Dict{Symbol,Vector{Float64}}()
        for var in var_syms
            inner[var] = Float64[]          # will push! into it
        end
        accum[meas] = inner
    end
    return accum
end

"Append values from `meas_one_rep` (1‑replicate output) into `accum`."
@inline function update_accumulator!(accum, meas_one_rep)
    for (meas, vdict) in meas_one_rep           # meas ⇒ variant ⇒ Vector(1)
        inner = accum[meas]
        for (var, vec1) in vdict
            push!(inner[var], vec1[1])          # vec1 has length 1
        end
    end
    return
end

"""
    apply_measures_for_year(profiles_year) -> OrderedDict

For each scenario and each `m`, apply the global polarization measures to all
replicates and accumulate distributions per variant.

Returns: `scenario ⇒ m ⇒ (measure ⇒ variant ⇒ Vector{Float64})`.
"""
function apply_measures_for_year(
    profiles_year::OrderedDict{String,<:Any}
)::OrderedDict{String,OrderedDict{Int,Dict{Symbol,Dict{Symbol,Vector{Float64}}}}}

    out = OrderedDict{String,OrderedDict{Int,Dict{Symbol,Dict{Symbol,Vector{Float64}}}}}()

    for (scen, m_map) in profiles_year
        scen_out = OrderedDict{Int,Dict{Symbol,Dict{Symbol,Vector{Float64}}}}()

        for (m, slice) in m_map            # slice :: ProfilesSlice
            @assert slice isa ProfilesSlice

            variants   = collect(keys(slice.paths))
            n_rep_max  = maximum(length(slice.paths[v]) for v in variants)

            #####  first replicate: discover measure names  #####
            var_map1 = Dict(var => [slice[var, 1]] for var in variants)
            decode_each!(var_map1)
            meas1     = apply_all_measures_to_bts(var_map1)
            meas_syms = collect(keys(meas1))

            accum = init_accumulator(variants, meas_syms)
            update_accumulator!(accum, meas1)

            #####  progress bar  #####
            prog = pm.Progress(n_rep_max - 1;  desc = "[$scen|m=$m]", barlen = 30)

            #####  remaining replicates  #####
            rep_counter = 1
            for i in 2:n_rep_max
                var_map = Dict{Symbol,Vector{DataFrame}}()

                for var in variants
                    length(slice.paths[var]) < i && continue
                    df = slice[var, i]
                    decode_profile_column!(df)
                    var_map[var] = [df]
                end
                isempty(var_map) && continue

                meas_i = apply_all_measures_to_bts(var_map)
                update_accumulator!(accum, meas_i)

                pm.next!(prog)                           # advance bar
                rep_counter += 1
                rep_counter % 10 == 0 && GC.gc()
            end
            pm.finish!(prog)

            scen_out[m] = accum
            GC.gc()
        end
        out[scen] = scen_out
    end
    return out
end

const GLOBAL_MEASURE_DIR = joinpath(INT_DIR, "global_measures")
mkpath(GLOBAL_MEASURE_DIR)   # ensure the directory exists

"""
    save_or_load_measures_for_year(year, profiles_year;
                                   dir       = GLOBAL_MEASURE_DIR,
                                   overwrite = false,
                                   verbose   = true)

For a single `year`:

- If `dir/measures_YEAR.jld2` exists and `overwrite == false`, emits a warning and loads `measures` from disk.
- Otherwise, runs `apply_measures_for_year(profiles_year)`, saves the result under the name `measures`, and returns it.
"""
function save_or_load_measures_for_year(year,
                                        profiles_year;
                                        dir::AbstractString = GLOBAL_MEASURE_DIR,
                                        overwrite::Bool     = false,
                                        verbose::Bool       = true)

    path = joinpath(dir, "measures_$(year).jld2")

    if isfile(path) && !overwrite
        verbose && @warn "Global measures for $year already cached at $path; loading."
        measures = nothing
        @load path measures
        return measures
    end

    verbose && @info "Computing global measures for year $year…"
    measures = apply_measures_for_year(profiles_year)
    @save path measures
    verbose && @info "Saved global measures for year $year → $path"
    return measures
end

const GROUP_DIR = joinpath(INT_DIR, "group_metrics"); mkpath(GROUP_DIR)

"""
    update_accum!(accum::Dict, res::Dict, variants)

Internal helper: append one‑replicate results `res` into aggregate `accum` for
each `variant`. Creates missing variant vectors on the fly.
"""
function update_accum!(accum::Dict, res::Dict, variants)
    for (met, vdict) in res
        inner = get!(accum, met) do
            # first time we see this metric → create inner dict with empty vectors
            Dict(var => Float64[] for var in variants)
        end
        for (var, vec1) in vdict          # vec1 length == 1
            push!(get!(inner, var, Float64[]), vec1[1])   # create variant slot if absent
        end
    end
end

# ─────────────────── streaming apply_group_metrics_for_year ───────────────────
"""
    apply_group_metrics_for_year_streaming(profiles_year, cfg) -> OrderedDict

Stream over all replicates to compute group metrics (`C`, `D`, optionally `G`)
for every scenario, `m`, and demographic in `cfg.demographics`.

Returns: `scenario ⇒ m ⇒ dem ⇒ metric ⇒ variant ⇒ Vector`.
"""
function apply_group_metrics_for_year_streaming(
        profiles_year::OrderedDict{String,<:Any},
        cfg)

    out = OrderedDict()

    for (scen, m_map) in profiles_year
        scen_out = OrderedDict()

        for (m, slice) in m_map             # slice :: ProfilesSlice
            variants   = collect(keys(slice.paths))
            n_rep_max  = maximum(length(slice.paths[v]) for v in variants)

            dem_out = OrderedDict()

            for dem in cfg.demographics
                dem_sym = dem isa Symbol ? dem : Symbol(dem)
                @info "  → scenario=$scen, m=$m, dem=$dem_sym"

                accum = Dict{Symbol,Dict{Symbol,Vector{Float64}}}()
                prog  = pm.Progress(n_rep_max; desc="[$scen|m=$m|$dem_sym]", barlen=28)

                for i in 1:n_rep_max
                    var_map = Dict{Symbol,Vector{DataFrame}}()
                    for var in variants
                        length(slice.paths[var]) < i && continue
                        df = slice[var, i]
                        decode_profile_column!(df)
                        var_map[var] = [df]
                    end
                    isempty(var_map) && (next!(prog); continue)

                    res = bootstrap_group_metrics(var_map, dem_sym)
                    update_accum!(accum, res, variants)
                    pm.next!(prog)
                end
                pm.finish!(prog)
                dem_out[dem_sym] = accum
                GC.gc()
            end
            scen_out[m] = dem_out
        end
        out[scen] = scen_out
    end
    return out          # scenario ⇒ m ⇒ dem ⇒ metric ⇒ variant ⇒ Vector
end

# directory layout for per‑DataFrame caches
"""
    _perdf(dir, year, scen, m, dem, rep) -> String

Compute a canonical cache path for metrics of a single DataFrame (replicate).
Layout: `dir/per_df/{year}/{scen}/m{m}/{dem}/rep{rep}.jld2`.
"""
_perdf(dir, year, scen, m, dem, rep) =
    joinpath(dir, "per_df", string(year), string(scen), "m$m", string(dem),
             "rep$rep.jld2")

"Ensure the parent directory of `path` exists."
_mkparent(path) = mkpath(dirname(path))

"""
    compute_and_cache_group_metrics_per_df!(year, profiles_year, cfg;
                                            dir=GROUP_DIR, overwrite=false,
                                            verbose=true)

Pass 1 of the two‑pass pipeline: compute group metrics for each replicate and
**write per‑DataFrame caches**. Safe to re‑run: respects `overwrite=false`.
"""
function compute_and_cache_group_metrics_per_df!(
        year::Int,
        profiles_year::OrderedDict{String,<:Any},
        cfg;
        dir::AbstractString = GROUP_DIR,
        overwrite::Bool     = false,
        verbose::Bool       = true)

    for (scen, m_map) in profiles_year
        for (m, slice) in m_map                 # slice :: ProfilesSlice
            variants   = collect(keys(slice.paths))
            n_rep_max  = maximum(length(slice.paths[v]) for v in variants)

            for dem in cfg.demographics
                dem_sym = Symbol(dem)
                pbar = pm.Progress(n_rep_max;
                                desc="[$year|$scen|m=$m|$dem_sym]",
                                barlen=28)

                for rep in 1:n_rep_max
                    cache_path = _perdf(dir, year, scen, m, dem_sym, rep)
                    if isfile(cache_path) && !overwrite
                        pm.next!(pbar); continue
                    end

                    var_map = Dict{Symbol,Vector{DataFrame}}()
                    for var in variants
                        length(slice.paths[var]) < rep && continue
                        df = slice[var, rep]                 # load
                        decode_profile_column!(df)
                        var_map[var] = [df]
                    end
                    isempty(var_map) && (pm.next!(pbar); continue)

                    res = bootstrap_group_metrics(var_map, dem_sym)

                    _mkparent(cache_path)
                    @save cache_path res              # ■ write to disk
                    pm.next!(pbar)
                end
                pm.finish!(pbar); GC.gc()
            end
        end
    end
end

# ────────────────── pass 2: aggregate caches ──────────────────
"""
    accumulate_cached_group_metrics_for_year!(year, profiles_year, cfg;
                                              dir=GROUP_DIR, verbose=true)

Pass 2 of the two‑pass pipeline: aggregate previously cached per‑DataFrame
metrics into distributions per `(scenario, m, demographic, metric, variant)`.
"""
function accumulate_cached_group_metrics_for_year!(
        year::Int,
        profiles_year::OrderedDict{String,<:Any},
        cfg;
        dir::AbstractString = GROUP_DIR,
        verbose::Bool       = true)

    out = OrderedDict()

    for (scen, m_map) in profiles_year
        scen_out = OrderedDict()
        for (m, slice) in m_map
            variants   = collect(keys(slice.paths))
            n_rep_max  = maximum(length(slice.paths[v]) for v in variants)

            dem_out = OrderedDict()
            for dem in cfg.demographics
                dem_sym = Symbol(dem)
                verbose && @info "  → aggregating $scen, m=$m, dem=$dem_sym"

                accum = Dict{Symbol,Dict{Symbol,Vector{Float64}}}()
                pbar  = pm.Progress(n_rep_max;
                                 desc="[$scen|m=$m|$dem_sym]", barlen=28)

                for rep in 1:n_rep_max
                    cache_path = _perdf(dir, year, scen, m, dem_sym, rep)
                    isfile(cache_path) || error("Missing cache $cache_path")
                    res = nothing; @load cache_path res
                    update_accum!(accum, res, variants)
                    pm.next!(pbar)
                end
                pm.finish!(pbar)
                dem_out[dem_sym] = accum
            end
            scen_out[m] = dem_out
        end
        out[scen] = scen_out
    end
    return out           # scenario ⇒ m ⇒ dem ⇒ metric ⇒ variant ⇒ Vector
end

# ────────────────── public API (drop‑in) ──────────────────
"""
    save_or_load_group_metrics_for_year(year, profiles_year, f3_entry;
                                        dir=GROUP_DIR, overwrite=false,
                                        two_pass=false, verbose=true)

Run the group‑metrics pipeline for one `year`. If `two_pass=true`, compute and
cache per‑DataFrame results first and then aggregate; otherwise stream
computation in a single pass. Always writes `group_metrics_YEAR.jld2` and
returns the aggregated object.
"""
function save_or_load_group_metrics_for_year(year::Int,
                                             profiles_year,
                                             f3_entry;
                                             dir::AbstractString = GROUP_DIR,
                                             overwrite::Bool     = false,
                                             two_pass::Bool      = false,
                                             verbose::Bool       = true)

    final_path = joinpath(dir, "group_metrics_$(year).jld2")

    if isfile(final_path) && !overwrite && !two_pass
        verbose && @warn "Group metrics for $year already cached; loading."
        metrics = nothing; @load final_path metrics; return metrics
    end

    if two_pass
        verbose && @info "Pass 1: computing & caching per‑DataFrame metrics…"
        compute_and_cache_group_metrics_per_df!(year, profiles_year, f3_entry.cfg;
                                                dir=dir, overwrite=overwrite,
                                                verbose=verbose)

        verbose && @info "Pass 2: aggregating cached metrics…"
        metrics = accumulate_cached_group_metrics_for_year!(year, profiles_year,
                                                            f3_entry.cfg;
                                                            dir=dir,
                                                            verbose=verbose)
    else
        verbose && @info "Computing group metrics for year $year (one‑pass)…"
        metrics = apply_group_metrics_for_year_streaming(profiles_year,
                                                         f3_entry.cfg)
    end

    @save final_path metrics
    verbose && @info "Saved aggregated metrics for $year → $final_path"
    return metrics
end



function describe_candidate_set(candidates::Vector{String})
    pretty_names = [join([uppercasefirst(lowercase(w)) for w in split(name, "_")], " ")
                    for name in candidates]
    return "Candidates: " * join(pretty_names, ", ")
end

function lines_alt_by_variant(measures_over_m::AbstractDict;
                              variants   = ["zero","random","mice"],
                              palette    = Makie.wong_colors(),
                              figsize    = (1000, 900),
                              candidate_label::String = "", year)

    ms        = sort(collect(keys(measures_over_m)))                # alt counts
    measures  = sort(collect(keys(first(values(measures_over_m))))) # :C, :D, …
    nv        = length(variants)

    # legend labels
    mlabels = Dict(
        :calc_reversal_HHI             => "HHI",
        :calc_total_reversal_component => "R",
        :fast_reversal_geometric       => "RHHI",
    )

    fig = Figure(resolution = figsize)
    rowgap!(fig.layout, 18)
    colgap!(fig.layout, 4)
    fig[1, 2] = GridLayout()                 # legend column
    colsize!(fig.layout, 2, 100)

    # header
    first_m, last_m = first(ms), last(ms)
    first_var = first(variants)
    n_bootstrap = length(first(values(first(values(measures_over_m))))[Symbol(first_var)])

    titlegrid = GridLayout(tellwidth = true)
    fig[0, 1] = titlegrid
    Label(titlegrid[1, 1];
          text = "Year = $(year)   •   Number of alternatives = $first_m … $last_m   •   $n_bootstrap pseudo-profiles",
          fontsize = 18, halign = :left)
    Label(titlegrid[2, 1];
          text = candidate_label,
          fontsize = 14, halign = :left)

    # axes
    axes = [Axis(fig[i, 1];
                 title   = variants[i],
                 xlabel  = "number of alternatives",
                 ylabel  = "value",
                 yticks  = (0:0.1:1, string.(0:0.1:1)),
                 xticks  = (ms, string.(ms)))
            for i in 1:nv]

    legend_handles = Lines[]
    legend_labels  = AbstractString[]

    # plot
    for (row, var) in enumerate(variants)
        ax = axes[row]

        for (j, meas) in enumerate(measures)
            col = palette[(j-1) % length(palette) + 1]

            meds = Float64[]          # medians per m
            q25s = Float64[]; q75s = Float64[]          # 25–75%
            p05s = Float64[]; p95s = Float64[]          # 5–95%

            for m in ms
                vals = measures_over_m[m][meas][Symbol(var)]
                push!(meds, median(vals))
                push!(q25s, quantile(vals, 0.25))
                push!(q75s, quantile(vals, 0.75))
                push!(p05s, quantile(vals, 0.05))
                push!(p95s, quantile(vals, 0.95))
            end

            # 90 % band (5–95)
            band!(ax, ms, p05s, p95s; color = (col, 0.12), linewidth = 0)

            # inter-quartile band (25–75)
            band!(ax, ms, q25s, q75s; color = (col, 0.25), linewidth = 0)

            # median line
            ln = lines!(ax, ms, meds;
                        color = col, linewidth = 2,
                        label = get(mlabels, meas, string(meas)))

            if row == 1
                push!(legend_handles, ln)
                push!(legend_labels, get(mlabels, meas, string(meas)))
            end
        end
    end

    Legend(fig[1:nv, 2], legend_handles, legend_labels)
    resize_to_layout!(fig)
    return fig
end




"""
    plot_scenario_year(year, scenario, f3, all_meas;
                       variant=\"mice\", palette, figsize) -> Figure

Convenience wrapper to produce a single‑scenario plot for `year` and
`scenario` using `lines_alt_by_variant`. Reconstructs the candidate set
exactly as in `generate_profiles_for_year`, warns if different sets were found
across replicates, and passes a human‑readable candidate label to the plotting
helper.
"""
function plot_scenario_year(
    year,
    scenario,
    f3,
    all_meas;
    variant = "mice",
    palette = Makie.wong_colors(),
    figsize = (500,400),
)
    # lookups
    f3_entry   = f3[year]
    cfg        = f3_entry.cfg

    reps_raw   = f3_entry.data
    year_meas  = all_meas[year]
    meas_map   = year_meas[scenario]
    scen_obj   = findfirst(s->s.name==scenario, cfg.scenarios)

    # recompute full candidate set exactly as in generate_profiles_for_year
    sets = unique(map(df->
        compute_candidate_set(df;
            candidate_cols = cfg.candidates,
            m              = cfg.max_candidates,
            force_include  = cfg.scenarios[scen_obj].candidates),
        reps_raw))
    if length(sets)!=1
        msg = "Year $year scenario $scenario: found $(length(sets)) distinct candidate sets; using first."
        @warn msg
        open(CANDLOG,"a") do io
            println(io,"[$(Dates.format(now(),"yyyy-mm-dd HH:MM:SS"))] $msg")
        end
    end
    full_list = sets[1]
    candidate_label = describe_candidate_set(full_list)

    # delegate to your Makie helper
    fig = lines_alt_by_variant(
        meas_map;
        variants        = [variant],
        palette         = palette,
        figsize         = figsize,
        year            = year,
        candidate_label = candidate_label,
    )
    return fig
end

# ──────────────────────────────────────────────────────────────────
# helper: replicate the candidate-set logic from generate_profiles…
# ──────────────────────────────────────────────────────────────────
"""
    _full_candidate_list(cfg, reps_raw, scen_obj) -> Vector{String}

Compute the unique candidate list used by `generate_profiles_for_year`,
warning and picking the first when multiple sets appear across bootstrap draws.
"""
function _full_candidate_list(cfg, reps_raw, scen_obj)
    sets = unique(map(df ->
        compute_candidate_set(df;
            candidate_cols = cfg.candidates,
            m              = cfg.max_candidates,
            force_include  = scen_obj.candidates),
        reps_raw))
    length(sets) != 1 && @warn "Multiple candidate sets; using first"
    return sets[1]
end

"""
    plot_group_demographics_lines(all_gm, f3, year, scenario;
                                  variants=[:zero,:random,:mice],
                                  measures=[:C,:D,:G], maxcols=3, n_yticks=5,
                                  palette=Makie.wong_colors(),
                                  clist_size=60, demographics=f3[year].cfg.demographics)
        -> Figure

One panel per demographic; **x = number of alternatives (m)**.
For every *(measure, variant)* pair the panel shows

* a translucent band between Q25 and Q75
* a line for the mean.

The title and candidate list match the original
`plot_group_demographics`, but the layout and styling come from
`lines_group_measures_over_m`.
"""
function plot_group_demographics_lines(
        all_gm,
        f3,
        year::Int,
        scenario::String;
        variants      = [:zero, :random, :mice],
        measures      = [:C, :D, :G],
        maxcols::Int  = 3,
        n_yticks::Int = 5,
        palette       = Makie.wong_colors(), clist_size = 60,
        demographics = f3[year].cfg.demographics
)

    # ── data slice & metadata ──────────────────────────────────────────
    gm            = all_gm[year][scenario]                  # m ⇒ (dem ⇒ …)
    m_values_int  = sort(collect(keys(gm)))                 # Vector{Int}
    xs_m          = Float32.(m_values_int)                  # Makie prefers Float32
    n_demo        = length(demographics)

    scenobj = only(filter(s->s.name==scenario, f3[year].cfg.scenarios))
    cand_lbl = describe_candidate_set(
                 _full_candidate_list(f3[year].cfg, f3[year].data, scenobj))

    # any slice gives the bootstrap length
    sample_slice = gm[first(m_values_int)][Symbol(first(demographics))]
    n_boot       = length(sample_slice[variants[1]][:C])

    # ── colour / style dictionaries ────────────────────────────────────
    measure_cols   = Dict(measures[i] => palette[i] for i in eachindex(measures))
    variant_styles = Dict(:zero => :solid, :random => :dash, :mice => :dot)

    # ── figure geometry ────────────────────────────────────────────────
    ncol       = min(maxcols, n_demo)
    nrow       = ceil(Int, n_demo / ncol)
    title_txt  = "Year $(year) • $(n_boot) pseudo-profiles • m = $(first(m_values_int)) … $(last(m_values_int))"
    fig_width  = max(300*ncol, 10*length(title_txt) + 60)   # widen if title is long
    fig_height = 300*nrow

    fig = Figure(resolution = (fig_width, fig_height))
    rowgap!(fig.layout, 24);  colgap!(fig.layout, 24)

    # headers
    fig[1, 1:ncol] = Label(fig, title_txt;  fontsize = 20, halign = :left)
    fig[2, 1:ncol] = Label(fig,  join(TextWrap.wrap("$cand_lbl"; width=clist_size)); fontsize = 14, halign = :left)
    header_rows = 2

    # legend collectors
    legend_handles = Any[]; legend_labels = String[]

    # ── main panels ────────────────────────────────────────────────────
    for (idx, demo) in enumerate(demographics)
        r, c = fldmod1(idx, ncol)
        ax = Axis(fig[r + header_rows, c];
                  title  = demo,
                  xlabel = "number of alternatives",
                  ylabel = "value",
                  xticks = (xs_m, string.(m_values_int)))            # avoid stretch

        allvals = Float32[]                        # for nice y-ticks

        for meas in measures, var in variants
            vals_per_m = map(m_values_int) do m
                v = gm[m][Symbol(demo)][var]
                arr = meas === :G ? sqrt.(v[:C] .* v[:D]) : v[meas]
                Float32.(arr)
            end

            append!(allvals, vcat(vals_per_m...))

            # meds32 = Float32.(mean.(vals_per_m))
            meds32 = Float32.(median.(vals_per_m))
            q25s32 = Float32.(map(x -> quantile(x, 0.25f0), vals_per_m))
            q75s32 = Float32.(map(x -> quantile(x, 0.75f0), vals_per_m))

            col = measure_cols[meas]
            sty = variant_styles[var]

            band!(ax, xs_m, q25s32, q75s32; color = (col, 0.20), linewidth = 0)
            ln = lines!(ax, xs_m, meds32;        color = col, linestyle = sty, linewidth = 2)

            if idx == 1
                push!(legend_handles, ln)
                push!(legend_labels, "$(meas) • $(var)")
            end
        end

        # tidy y-ticks
        y_min, y_max = extrema(allvals)
        ticks  = collect(range(y_min, y_max; length = n_yticks))
        ax.yticks[] = (ticks, string.(round.(ticks; digits = 3)))
    end

    # ── legend column ──────────────────────────────────────────────────
    Legend(fig[header_rows+1 : header_rows+nrow, ncol+1],
           legend_handles, legend_labels; tellheight = false)

    # now that the legend is placed, column ncol+1 exists → shrink it
    colsize!(fig.layout, ncol + 1, Relative(0.25))

    resize_to_layout!(fig)
    return fig
end

"""
    save_plot(fig, year, scenario, cfg; variant, dir = "imgs", ext = ".png")

Save `fig` under `dir/`, creating the directory if needed.
The file name pattern is:

    {year}_{scenario}_{variant}_B{n_bootstrap}_M{max_m}_{yyyymmdd-HHMMSS}{ext}
"""
function save_plot(fig, year::Int, scenario::AbstractString, cfg;
                   variant::AbstractString,
                   dir::AbstractString = "imgs",
                   ext::AbstractString = ".png")

    # 1. make sure the directory exists
    mkpath(dir)

    # 2. assemble a human-readable, reproducible file name
    time_stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    max_m      = maximum(cfg.m_values_range)
    fname      = joinpath(dir,
        string(year, '_', scenario, '_', variant,
               "_B", cfg.n_bootstrap,
               "_M", max_m,
               '_', time_stamp, ext))

    # 3. save
    save(fname, fig; px_per_unit = 4)
    @info "saved plot → $fname"
    return fname
end
