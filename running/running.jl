cd("..")

using Revise
using PrefPol
import PrefPol as pp

# ------------------------------------------------------------------
# Bootstraps: save (idempotent) and load
# ------------------------------------------------------------------
saved_bootstrap_paths = pp.save_all_bootstraps()
saved_bootstrap_paths = nothing

bootstrap_index = pp.load_all_bootstraps()   # year ⇒ (data, cfg, path)

# ------------------------------------------------------------------
# Imputation indices (idempotent unless overwrite=true)
# ------------------------------------------------------------------
imputed_index_paths = pp.impute_from_f3(bootstrap_index; overwrite = false)
imputed_index_paths = nothing

# ------------------------------------------------------------------
# Load per‑year imputation indices
# ------------------------------------------------------------------
imputed_year_2006 = pp.load_imputed_year(2006)
imputed_year_2018 = pp.load_imputed_year(2018)
imputed_year_2022 = pp.load_imputed_year(2022)

# ------------------------------------------------------------------
# Generate streamed profiles per year
# ------------------------------------------------------------------
profiles_2006 = pp.generate_profiles_for_year_streamed_from_index(
                    2006, bootstrap_index[2006], imputed_year_2006; overwrite = false)

profiles_2018 = pp.generate_profiles_for_year_streamed_from_index(
                    2018, bootstrap_index[2018], imputed_year_2018; overwrite = false)

profiles_2022 = pp.generate_profiles_for_year_streamed_from_index(
                    2022, bootstrap_index[2022], imputed_year_2022; overwrite = false)

# ------------------------------------------------------------------
# Global measures (per year)
# ------------------------------------------------------------------
measures_2006 = pp.save_or_load_measures_for_year(2006, profiles_2006;
                    overwrite = false,   # set true to rebuild
                    verbose   = true)    # progress / info logs

measures_2018 = pp.save_or_load_measures_for_year(2018, profiles_2018;
                    overwrite = false,   # set true to rebuild
                    verbose   = true)    # progress / info logs

measures_2022 = pp.save_or_load_measures_for_year(2022, profiles_2022;
                    overwrite = false,   # set true to rebuild
                    verbose   = true)    # progress / info logs

# ------------------------------------------------------------------
# Group metrics (per year)
# ------------------------------------------------------------------
group_metrics_2006 = pp.save_or_load_group_metrics_for_year(
                        2006, profiles_2006, bootstrap_index[2006];
                        overwrite = false, verbose = true, two_pass = true)

group_metrics_2018 = pp.save_or_load_group_metrics_for_year(
                        2018, profiles_2018, bootstrap_index[2018];
                        overwrite = false, verbose = true, two_pass = true)

group_metrics_2022 = pp.save_or_load_group_metrics_for_year(
                        2022, profiles_2022, bootstrap_index[2022];
                        overwrite = false, verbose = true, two_pass = true)

# ------------------------------------------------------------------
# Scenario plots — 2022 (Lula vs Bolsonaro)
# ------------------------------------------------------------------
plot_measures_2022 = Dict(2022 => measures_2022)

fig_2022_mice   = pp.plot_scenario_year(2022, "lula_bolsonaro", bootstrap_index, plot_measures_2022; variant = "mice")


fig_2022_random = pp.plot_scenario_year(2022, "lula_bolsonaro", bootstrap_index, plot_measures_2022; variant = "random")

fig_2022_zero   = pp.plot_scenario_year(2022, "lula_bolsonaro", bootstrap_index, plot_measures_2022; variant = "zero")

cfg_2022 = bootstrap_index[2022].cfg
pp.save_plot(fig_2022_mice,   2022, "lula_bolsonaro", cfg_2022; variant = "mice")
pp.save_plot(fig_2022_random, 2022, "lula_bolsonaro", cfg_2022; variant = "random")
pp.save_plot(fig_2022_zero,   2022, "lula_bolsonaro", cfg_2022; variant = "zero")

# ------------------------------------------------------------------
# Scenario plots — 2006 (Lula vs Alckmin)
# ------------------------------------------------------------------
plot_measures_2006 = Dict(2006 => measures_2006)

fig_2006_mice   = pp.plot_scenario_year(2006, "lula_alckmin", bootstrap_index, plot_measures_2006; variant = "mice")
fig_2006_random = pp.plot_scenario_year(2006, "lula_alckmin", bootstrap_index, plot_measures_2006; variant = "random")
fig_2006_zero   = pp.plot_scenario_year(2006, "lula_alckmin", bootstrap_index, plot_measures_2006; variant = "zero")

cfg_2006 = bootstrap_index[2006].cfg
pp.save_plot(fig_2006_mice,   2006, "lula_alckmin", cfg_2006; variant = "mice")
pp.save_plot(fig_2006_random, 2006, "lula_alckmin", cfg_2006; variant = "random")
pp.save_plot(fig_2006_zero,   2006, "lula_alckmin", cfg_2006; variant = "zero")

# ------------------------------------------------------------------
# Scenario plots — 2018 (three scenarios)
# ------------------------------------------------------------------
plot_measures_2018 = Dict(2018 => measures_2018)

fig_2018_main_four     = pp.plot_scenario_year(2018, "main_four",      bootstrap_index, plot_measures_2018; variant = "mice")
fig_2018_no_forcing    = pp.plot_scenario_year(2018, "no_forcing",     bootstrap_index, plot_measures_2018; variant = "mice")
fig_2018_lula_bolsonaro = pp.plot_scenario_year(2018, "lula_bolsonaro", bootstrap_index, plot_measures_2018; variant = "mice")

cfg_2018 = bootstrap_index[2018].cfg
pp.save_plot(fig_2018_main_four,      2018, "main_four",     cfg_2018; variant = "mice")
pp.save_plot(fig_2018_no_forcing,     2018, "no_forcing",    cfg_2018; variant = "mice")
pp.save_plot(fig_2018_lula_bolsonaro, 2018, "lula_bolsonaro", cfg_2018; variant = "mice")

# ------------------------------------------------------------------
# Group-demographics panels — 2018
# ------------------------------------------------------------------
fig_2018_main_four_dem_main = pp.plot_group_demographics_lines(
    Dict(2018 => group_metrics_2018), bootstrap_index, 2018, "main_four";
    variants = [:mice], maxcols = 2, clist_size = 60,
    demographics = ["Income", "Ideology"])

fig_2018_main_four_dem_other = pp.plot_group_demographics_lines(
    Dict(2018 => group_metrics_2018), bootstrap_index, 2018, "main_four";
    variants = [:mice], maxcols = 3, clist_size = 60,
    demographics = setdiff(bootstrap_index[2018].cfg.demographics, ["Income", "Ideology"]))

pp.save_plot(fig_2018_main_four_dem_other, 2018, "main_four_group_therest", cfg_2018; variant = "mice")

# ------------------------------------------------------------------
# Group-demographics panels — 2022 (split main vs others)
# ------------------------------------------------------------------
dems_main_2022 = ["Ideology", "PT", "Abortion", "Religion", "Sex", "Income"]

fig_2022_lula_bolsonaro_dem_main = pp.plot_group_demographics_lines(
    Dict(2022 => group_metrics_2022), bootstrap_index, 2022, "lula_bolsonaro";
    variants = [:mice], maxcols = 3, clist_size = 60, demographics = dems_main_2022)

pp.save_plot(fig_2022_lula_bolsonaro_dem_main, 2022, "lula_bolsonaro_main", bootstrap_index[2022].cfg; variant = "mice")

fig_2022_lula_bolsonaro_dem_other = pp.plot_group_demographics_lines(
    Dict(2022 => group_metrics_2022), bootstrap_index, 2022, "lula_bolsonaro";
    variants = [:mice], maxcols = 3, clist_size = 60,
    demographics = setdiff(bootstrap_index[2022].cfg.demographics, dems_main_2022))

pp.save_plot(fig_2022_lula_bolsonaro_dem_other, 2022, "lula_bolsonaro_others", bootstrap_index[2022].cfg; variant = "mice")

# ------------------------------------------------------------------
# Group-demographics panels — 2006 (all demographics)
# ------------------------------------------------------------------
fig_2006_lula_alckmin_group = pp.plot_group_demographics_lines(
    Dict(2006 => group_metrics_2006), bootstrap_index, 2006, "lula_alckmin";
    variants = [:mice], maxcols = 3, clist_size = 60)

pp.save_plot(fig_2006_lula_alckmin_group, 2006, "lula_alckmin_group", cfg_2006; variant = "mice")
