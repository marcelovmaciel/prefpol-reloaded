#### preferences_display.jl — Views & pretty-printing (UI-only code)


# Depends on types:
#   CandidatePool, CandidateId, WeakRank, StrictRank, ordered_candidates

"Pretty print wrapper that renders a StrictRank using candidate names."
struct StrictRankView{N}
    r::StrictRank{N}
    pool::CandidatePool{N}
end

"Return a wrapper that customizes `show` to print candidate names in rank order."
pretty(r::StrictRank{N}, pool::CandidatePool{N}) where {N} = StrictRankView{N}(r, pool)

function Base.show(io::IO, v::StrictRankView{N}) where {N}
    names = ordered_candidates(v.r, v.pool)
    print(io, "StrictRank(", join(string.(names), " ≻ "), ")")
end

"View wrapper for weak orders (levels of indifference)."
struct WeakOrderView{N}
    levels::Vector{Vector{CandidateId}}
    pool::CandidatePool{N}
    hide_unranked::Bool
end

"Pretty wrapper for weak-order levels."
function pretty(levels::Vector{Vector{CandidateId}}, pool::CandidatePool{N}; hide_unranked::Bool=false) where {N}
    WeakOrderView{N}(levels, pool, hide_unranked)
end

function Base.show(io::IO, v::WeakOrderView{N}) where {N}
    levels = v.levels
    omitted = Symbol[]
    if v.hide_unranked && !isempty(levels)
        un = levels[end]
        if !isempty(un)
            append!(omitted, [v.pool.names[Int(i)] for i in un])
        end
        levels = levels[1:end-1]
    end
    level_strs = String[]
    for ids in levels
        syms = [v.pool.names[Int(i)] for i in ids]
        push!(level_strs, join(string.(syms), " ~ "))
    end
    print(io, "WeakOrder(", join(level_strs, " ≻ "), ")")
    if !isempty(omitted)
        print(io, " (unranked omitted: ", join(string.(omitted), ", "), ")")
    end
end

# ---------- Dodgson/pairwise colored table (terminal UI) ----------

# Robust label extractor from either `labels` kw or a CandidatePool `pool`
function _labels_from(pool, n)
    if pool === nothing
        return ["A$i" for i in 1:n]
    end
    # Try common field names
    for fld in (:names, :labels, :cands, :candidates, :display, :codes, :ids)
        if hasproperty(pool, fld)
            v = getfield(pool, fld)
            if v !== nothing
                xs = String.(v)
                xs = xs[1:min(end,n)]
                if length(xs) < n
                    append!(xs, ["A$(i)" for i in length(xs)+1:n])
                end
                return xs
            end
        end
    end
    # Try callable accessors like pool.name(i)
    for f in (:name, :label, :display)
        if hasproperty(pool, f)
            g = getfield(pool, f)
            if g isa Function
                return [String(g(i)) for i in 1:n]
            end
        end
    end
    return ["A$i" for i in 1:n]
end

"""
show_dodgson_table_color(pairwise; labels=nothing, pool=nothing)

Colored Dodgson/pairwise table for the terminal.
Pass `pool=<CandidatePool>` to pull labels, or `labels=[...]` to override.
"""
function show_dodgson_table_color(pairwise; labels=nothing, pool=nothing)
    M = hasproperty(pairwise, :matrix) ? getfield(pairwise, :matrix) : pairwise
    n = size(M,1); @assert n == size(M,2)
    lbls = labels === nothing ? _labels_from(pool, n) : String.(labels)
    if length(lbls) != n
        lbls = lbls[1:min(end,n)]
        if length(lbls) < n
            append!(lbls, ["A$(i)" for i in length(lbls)+1:n])
        end
    end

    fmt = (v,i,j) -> i==j ? "–" : ismissing(v) ? "·" : v==1 ? ">" : v==-1 ? "<" : "≃"

    h_win  = Highlighter(f=(d,i,j)->i!=j && !ismissing(d[i,j]) && d[i,j]== 1, crayon=Crayon(foreground=:green,  bold=true))
    h_loss = Highlighter(f=(d,i,j)->i!=j && !ismissing(d[i,j]) && d[i,j]==-1, crayon=Crayon(foreground=:red))
    h_tie  = Highlighter(f=(d,i,j)->i!=j && !ismissing(d[i,j]) && d[i,j]== 0, crayon=Crayon(foreground=:yellow))
    h_miss = Highlighter(f=(d,i,j)->i!=j &&  ismissing(d[i,j]),                 crayon=Crayon(faint=true))
    h_diag = Highlighter(f=(d,i,j)->i==j,                                        crayon=Crayon(faint=true))

    pretty_table(M; header = lbls, row_labels = lbls, row_label_column_title = "",
        formatters = fmt, alignment = :c, tf = tf_unicode_rounded,
        highlighters = (h_win, h_loss, h_tie, h_miss, h_diag),
    )

    println("\nLegend: ", Crayon(foreground=:green, bold=true)("> = win"), " ",
        Crayon(foreground=:red)("< = loss"), " ",
        Crayon(foreground=:yellow)("≃ = tie"), " ",
        Crayon(faint=true)("· = missing, – = diagonal"),
    )
end
